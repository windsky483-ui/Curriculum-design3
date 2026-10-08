/* STM32F407 + MPU6050：PB8/PB9=I2C1，PA9/PA10=USART1，PF9=DS0，PF8=蜂鸣器。 */

#include "stm32f4xx.h"
#include <stdio.h>
#include "reco.h"          /* 动作识别：特征提取 + 随机森林推理 */

#define MPU6050_ADDR        0x68U   /* AD0 接地 => 0x68；若 AD0 接高电平则为 0x69 */

#define MPU6050_SMPLRT_DIV  0x19U
#define MPU6050_CONFIG      0x1AU
#define MPU6050_GYRO_CONFIG 0x1BU
#define MPU6050_ACCEL_CONFIG 0x1CU
#define MPU6050_ACCEL_XOUT_H 0x3BU
#define MPU6050_PWR_MGMT_1  0x6BU
#define MPU6050_WHO_AM_I    0x75U

/* ACCEL_CONFIG/GYRO_CONFIG 均设置为 0x08：±4g、±500dps。 */
#define ACC_LSB_PER_G       8192
#define GYRO_LSB_PER_DPS    655      /* 65.5 * 10，用整数运算避免浮点 */

#define OUTPUT_CSV              1

#define MPU_SAMPLE_PERIOD_MS    10U
#define MPU_LED_TOGGLE_SAMPLES  50U

#define INV_LSB_PER_G       (1.0f / (float)ACC_LSB_PER_G)
#define INV_LSB_PER_DPS     (10.0f / (float)GYRO_LSB_PER_DPS)

#define I2C_TIMEOUT         200000U

static volatile uint32_t s_tick_ms = 0;

void SysTick_Handler(void)
{
    s_tick_ms++;
}

static void systick_init(void)
{
    SysTick->LOAD = 16000U - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk |
                    SysTick_CTRL_TICKINT_Msk   |
                    SysTick_CTRL_ENABLE_Msk;
}

static void delay_ms(uint32_t ms)
{
    uint32_t start = s_tick_ms;
    while ((s_tick_ms - start) < ms) {
    }
}

static void clock_hsi_16m(void)
{
    RCC->CR |= RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0U) {
    }

    RCC->CFGR &= ~RCC_CFGR_SW;
    while ((RCC->CFGR & RCC_CFGR_SWS) != 0U) {
    }

    RCC->CR &= ~RCC_CR_PLLON;
    RCC->CFGR &= ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2);

    FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_PRFTEN;
}

static void uart1_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;

    GPIOA->MODER &= ~((3U << (9 * 2)) | (3U << (10 * 2)));
    GPIOA->MODER |= ((2U << (9 * 2)) | (2U << (10 * 2)));
    GPIOA->OTYPER &= ~((1U << 9) | (1U << 10));
    GPIOA->OSPEEDR |= (3U << (9 * 2)) | (3U << (10 * 2));
    GPIOA->PUPDR &= ~((3U << (9 * 2)) | (3U << (10 * 2)));
    GPIOA->PUPDR |= (1U << (10 * 2));
    GPIOA->AFR[1] &= ~((0xFU << 4) | (0xFU << 8));
    GPIOA->AFR[1] |= ((7U << 4) | (7U << 8));

    USART1->CR1 = 0U;
    USART1->BRR = (16000000U + 115200U / 2U) / 115200U;
    USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

static void uart_putc(char c)
{
    while ((USART1->SR & USART_SR_TXE) == 0U) {
    }
    USART1->DR = (uint8_t)c;
}

int fputc(int ch, FILE *f)
{
    (void)f;
    uart_putc((char)ch);
    return ch;
}

#define LED_DS0_ON()      (GPIOF->BSRR = GPIO_BSRR_BR_9)   /* 低电平点亮 */
#define LED_DS0_OFF()     (GPIOF->BSRR = GPIO_BSRR_BS_9)
#define LED_DS0_TOGGLE()  (GPIOF->ODR ^= (1U << 9))

static void led_ds0_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOFEN;

    GPIOF->MODER &= ~(3U << (9 * 2));
    GPIOF->MODER |= (1U << (9 * 2));
    GPIOF->OTYPER &= ~(1U << 9);
    GPIOF->OSPEEDR |= (2U << (9 * 2));
    GPIOF->PUPDR &= ~(3U << (9 * 2));

    LED_DS0_OFF();
}

#define BEEP_PIN     8U
#define BEEP_ON()    (GPIOF->BSRR = GPIO_BSRR_BS_8)
#define BEEP_OFF()   (GPIOF->BSRR = GPIO_BSRR_BR_8)

static void beep_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOFEN;
    GPIOF->MODER &= ~(3U << (BEEP_PIN * 2));
    GPIOF->MODER |= (1U << (BEEP_PIN * 2));
    GPIOF->OTYPER &= ~(1U << BEEP_PIN);
    GPIOF->OSPEEDR |= (2U << (BEEP_PIN * 2));
    GPIOF->PUPDR &= ~(3U << (BEEP_PIN * 2));
    BEEP_OFF();
}

#define RECO_HOP_SAMPLES     25U      /* 每 0.25 秒推理一次（100Hz / 25） */
#define RECO_FALL_CONFIRM    3U       /* 连续 3 个窗口判跌倒才报警（防误报） */
#define RECO_ALARM_MS        5000U    /* 蜂鸣器响 5 秒后自动停 */

#define RECO_FALL_ALARM_ENABLE   0

static float s_acc_win[RECO_WIN_SAMPLES][3];
static float s_gyro_win[RECO_WIN_SAMPLES][3];
static float s_acc_lin[RECO_WIN_SAMPLES][3];
static float s_gyro_lin[RECO_WIN_SAMPLES][3];
static float s_feat[RECO_FEATURES];
static float s_proba[RECO_CLASSES];
static uint16_t s_win_pos = 0U;
static uint16_t s_win_fill = 0U;
static uint16_t s_hop_cnt = 0U;
static uint8_t  s_fall_cnt = 0U;
static uint32_t s_alarm_until = 0U;
static uint8_t  s_feat_dump_done = 0U;

static const char *const RECO_CLASS_NAME[RECO_CLASSES] = {"still", "walk", "run", "fall"};

static void reco_push(int16_t ax, int16_t ay, int16_t az,
                      int16_t gx, int16_t gy, int16_t gz)
{
    s_acc_win[s_win_pos][0] = (float)ax * INV_LSB_PER_G;
    s_acc_win[s_win_pos][1] = (float)ay * INV_LSB_PER_G;
    s_acc_win[s_win_pos][2] = (float)az * INV_LSB_PER_G;
    s_gyro_win[s_win_pos][0] = (float)gx * INV_LSB_PER_DPS;
    s_gyro_win[s_win_pos][1] = (float)gy * INV_LSB_PER_DPS;
    s_gyro_win[s_win_pos][2] = (float)gz * INV_LSB_PER_DPS;

    s_win_pos = (uint16_t)((s_win_pos + 1U) % RECO_WIN_SAMPLES);
    if (s_win_fill < RECO_WIN_SAMPLES) {
        s_win_fill++;
    }
}

static void reco_step(void)
{
    uint16_t i;
    int cls;

    s_hop_cnt++;
    if ((s_hop_cnt < RECO_HOP_SAMPLES) || (s_win_fill < RECO_WIN_SAMPLES)) {
        return;
    }
    s_hop_cnt = 0U;

    /* 环形缓冲按时间顺序展开。 */
    for (i = 0U; i < RECO_WIN_SAMPLES; i++) {
        uint16_t k = (uint16_t)((s_win_pos + i) % RECO_WIN_SAMPLES);
        s_acc_lin[i][0] = s_acc_win[k][0];
        s_acc_lin[i][1] = s_acc_win[k][1];
        s_acc_lin[i][2] = s_acc_win[k][2];
        s_gyro_lin[i][0] = s_gyro_win[k][0];
        s_gyro_lin[i][1] = s_gyro_win[k][1];
        s_gyro_lin[i][2] = s_gyro_win[k][2];
    }

    reco_features(s_acc_lin, s_gyro_lin, RECO_WIN_SAMPLES, s_feat);
    cls = reco_predict(s_feat, s_proba);

    /* 上位机可忽略 F 行；保留一次特征输出便于现场排查。 */
    if (s_feat_dump_done == 0U) {
        uint16_t q;
        s_feat_dump_done = 1U;
        printf("F");
        for (q = 0U; q < RECO_FEATURES; q++) {
            printf(",%.4f", (double)s_feat[q]);
        }
        printf("\r\n");
    }

    printf("R,%lu,%d,%d,%s\r\n", (unsigned long)s_tick_ms, cls,
           (int)((s_proba[cls] * 100.0f) + 0.5f), RECO_CLASS_NAME[cls]);

#if RECO_FALL_ALARM_ENABLE
    /* 连续多个窗口判跌倒才报警，降低误报。 */
    if (cls == RECO_CLASS_FALL) {
        if (s_fall_cnt < 250U) {
            s_fall_cnt++;
        }
    } else {
        s_fall_cnt = 0U;
    }
    if ((s_fall_cnt >= RECO_FALL_CONFIRM) && (s_tick_ms >= s_alarm_until)) {
        s_alarm_until = s_tick_ms + RECO_ALARM_MS;
        printf("ALARM,%lu\r\n", (unsigned long)s_tick_ms);
    }
#else
    (void)s_fall_cnt;
    (void)s_alarm_until;
    (void)cls;
#endif
}

static void i2c1_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
    RCC->APB1ENR |= RCC_APB1ENR_I2C1EN;

    GPIOB->MODER &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOB->MODER |= ((2U << (8 * 2)) | (2U << (9 * 2)));
    GPIOB->OTYPER |= (1U << 8) | (1U << 9);
    GPIOB->OSPEEDR |= (3U << (8 * 2)) | (3U << (9 * 2));
    GPIOB->PUPDR &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOB->PUPDR |= (1U << (8 * 2)) | (1U << (9 * 2));
    GPIOB->AFR[1] &= ~((0xFU << 0) | (0xFU << 4));
    GPIOB->AFR[1] |= ((4U << 0) | (4U << 4));

    I2C1->CR1 = I2C_CR1_SWRST;
    I2C1->CR1 = 0U;
    delay_ms(1);
    I2C1->CR2 = 16U;
    I2C1->OAR1 = 0x4000U;
    I2C1->CCR = 80U;
    I2C1->TRISE = 17U;
    I2C1->CR1 = I2C_CR1_PE;
}

static uint8_t i2c_wait_flag(uint32_t mask)
{
    uint32_t t = I2C_TIMEOUT;
    while (((I2C1->SR1 & mask) == 0U) && (--t != 0U)) {
    }
    return (t != 0U) ? 1U : 0U;
}

static uint8_t i2c_start(void)
{
    I2C1->CR1 |= I2C_CR1_START;
    return i2c_wait_flag(I2C_SR1_SB);
}

static uint8_t i2c_send_addr(uint8_t addr7, uint8_t read)
{
    uint32_t t = I2C_TIMEOUT;

    I2C1->DR = (uint8_t)((addr7 << 1) | (read ? 1U : 0U));

    while (((I2C1->SR1 & (I2C_SR1_ADDR | I2C_SR1_AF)) == 0U) && (--t != 0U)) {
    }
    if ((t == 0U) || ((I2C1->SR1 & I2C_SR1_AF) != 0U)) {
        I2C1->SR1 &= ~I2C_SR1_AF;
        return 0U;
    }
    (void)I2C1->SR1;
    (void)I2C1->SR2;
    return 1U;
}

static uint8_t i2c_write_byte(uint8_t dat)
{
    if (i2c_wait_flag(I2C_SR1_TXE) == 0U) {
        return 0U;
    }
    I2C1->DR = dat;
    return 1U;
}

static uint8_t i2c_wait_btf(void)
{
    return i2c_wait_flag(I2C_SR1_BTF);
}

static void i2c_stop(void)
{
    I2C1->CR1 |= I2C_CR1_STOP;
    uint32_t t = I2C_TIMEOUT;
    while (((I2C1->SR2 & I2C_SR2_BUSY) != 0U) && (--t != 0U)) {
    }
}

static uint8_t i2c_probe(uint8_t addr7)
{
    uint8_t ok;
    if (i2c_start() == 0U) {
        return 0U;
    }
    ok = i2c_send_addr(addr7, 0U);
    i2c_stop();
    return ok;
}

static uint8_t mpu_read(uint8_t reg, uint8_t *buf, uint8_t len)
{
    if (i2c_start() == 0U) {
        return 0U;
    }
    if (i2c_send_addr(MPU6050_ADDR, 0U) == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_write_byte(reg) == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_wait_btf() == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_start() == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_send_addr(MPU6050_ADDR, 1U) == 0U) {
        i2c_stop();
        return 0U;
    }

    while (len > 0U) {
        if (len == 1U) {
            I2C1->CR1 &= ~I2C_CR1_ACK;
            I2C1->CR1 |= I2C_CR1_STOP;
        } else {
            I2C1->CR1 |= I2C_CR1_ACK;
        }
        if (i2c_wait_flag(I2C_SR1_RXNE) == 0U) {
            break;
        }
        *buf++ = (uint8_t)I2C1->DR;
        len--;
    }
    return (len == 0U) ? 1U : 0U;
}

static uint8_t mpu_write(uint8_t reg, uint8_t dat)
{
    if (i2c_start() == 0U) {
        return 0U;
    }
    if (i2c_send_addr(MPU6050_ADDR, 0U) == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_write_byte(reg) == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_write_byte(dat) == 0U) {
        i2c_stop();
        return 0U;
    }
    if (i2c_wait_btf() == 0U) {
        i2c_stop();
        return 0U;
    }
    i2c_stop();
    return 1U;
}

#if !OUTPUT_CSV
static void print_g(int32_t mg)
{
    if (mg < 0) {
        uart_putc('-');
        mg = -mg;
    }
    printf("%ld.%03ld", (long)(mg / 1000), (long)(mg % 1000));
}

static void print_dps(int32_t dps100)
{
    if (dps100 < 0) {
        uart_putc('-');
        dps100 = -dps100;
    }
    printf("%ld.%02ld", (long)(dps100 / 100), (long)(dps100 % 100));
}

static void print_c(int32_t c100)
{
    if (c100 < 0) {
        uart_putc('-');
        c100 = -c100;
    }
    printf("%ld.%02ld", (long)(c100 / 100), (long)(c100 % 100));
}
#endif /* !OUTPUT_CSV */

static void i2c_scan(void)
{
    uint8_t addr;
    printf("I2C 总线扫描结果：");
    for (addr = 0x08U; addr <= 0x77U; addr++) {
        if (i2c_probe(addr) != 0U) {
            printf("0x%02X ", addr);
        }
    }
    printf("\r\n");
}

static uint8_t mpu6050_init(void)
{
    if (mpu_write(MPU6050_PWR_MGMT_1, 0x80U) == 0U) {
        return 0U;
    }
    delay_ms(100);
    if (mpu_write(MPU6050_PWR_MGMT_1, 0x01U) == 0U) {
        return 0U;
    }
    delay_ms(10);
    if (mpu_write(MPU6050_SMPLRT_DIV, 0x09U) == 0U) {
        return 0U;
    }
    if (mpu_write(MPU6050_CONFIG, 0x03U) == 0U) {
        return 0U;
    }
    if (mpu_write(MPU6050_GYRO_CONFIG, 0x08U) == 0U) {
        return 0U;
    }
    if (mpu_write(MPU6050_ACCEL_CONFIG, 0x08U) == 0U) {
        return 0U;
    }
    return 1U;
}

int main(void)
{
    uint8_t buf[14];
    uint8_t who = 0U;
    uint32_t cnt = 0U;
    uint32_t err_cnt = 0U;
    uint32_t next_ms = 0U;

    clock_hsi_16m();
    systick_init();
    uart1_init();
    led_ds0_init();
    beep_init();
    i2c1_init();

    printf("\r\n==============================================\r\n");
    printf(" 普中-天马 F407 板载 MPU6050 测试程序\r\n");
    printf(" SCL=PB8  SDA=PB9  INT->PC0  地址=0x68\r\n");
#if RECO_FALL_ALARM_ENABLE
    printf(" 动作识别：静止/走路/跑步/跌倒（连续3个窗口判跌倒则蜂鸣器响）\r\n");
#else
    printf(" 动作识别：静止/走路/跑步/跌倒（本地推理，结果每0.25秒回传）\r\n");
#endif
    printf("==============================================\r\n");

    i2c_scan();

    if ((mpu_read(MPU6050_WHO_AM_I, &who, 1U) == 0U) || (who != 0x68U)) {
        printf("MPU6050 自检失败：WHO_AM_I = 0x%02X（应为 0x68）\r\n", who);
        printf("DS0 常亮表示六轴模块异常，请检查 PB8/PB9 接线、上拉和模块供电。\r\n");
        LED_DS0_ON();
        while (1) {
        }
    }
    printf("WHO_AM_I = 0x%02X （正常应为 0x68）\r\n", who);

    if (mpu6050_init() == 0U) {
        printf("MPU6050 初始化失败，DS0 常亮。\r\n");
        LED_DS0_ON();
        while (1) {
        }
    }
    printf("MPU6050 初始化完成，开始输出数据……\r\n\r\n");
    printf("DS0(PF9) 会随数据读取成功以 1Hz 闪烁 = 六轴正在运行。\r\n\r\n");
    delay_ms(200);
    next_ms = s_tick_ms;

    while (1) {
        next_ms += MPU_SAMPLE_PERIOD_MS;

        if (mpu_read(MPU6050_ACCEL_XOUT_H, buf, 14U) != 0U) {
            int16_t ax = (int16_t)(((uint16_t)buf[0] << 8) | buf[1]);
            int16_t ay = (int16_t)(((uint16_t)buf[2] << 8) | buf[3]);
            int16_t az = (int16_t)(((uint16_t)buf[4] << 8) | buf[5]);
            int16_t tp = (int16_t)(((uint16_t)buf[6] << 8) | buf[7]);
            int16_t gx = (int16_t)(((uint16_t)buf[8] << 8) | buf[9]);
            int16_t gy = (int16_t)(((uint16_t)buf[10] << 8) | buf[11]);
            int16_t gz = (int16_t)(((uint16_t)buf[12] << 8) | buf[13]);

#if OUTPUT_CSV
            printf("%lu,%lu,%d,%d,%d,%d,%d,%d,%d\r\n",
                   (unsigned long)cnt, (unsigned long)s_tick_ms,
                   ax, ay, az, tp, gx, gy, gz);
#else
            int32_t ax_mg = (int32_t)ax * 1000 / ACC_LSB_PER_G;
            int32_t ay_mg = (int32_t)ay * 1000 / ACC_LSB_PER_G;
            int32_t az_mg = (int32_t)az * 1000 / ACC_LSB_PER_G;
            int32_t gx_100 = (int32_t)gx * 1000 / GYRO_LSB_PER_DPS;
            int32_t gy_100 = (int32_t)gy * 1000 / GYRO_LSB_PER_DPS;
            int32_t gz_100 = (int32_t)gz * 1000 / GYRO_LSB_PER_DPS;
            int32_t tp_100 = (int32_t)tp * 100 / 340 + 3653;

            printf("[%lu] 原始: %6d %6d %6d | %6d | %6d %6d %6d\r\n",
                   (unsigned long)cnt, ax, ay, az, tp, gx, gy, gz);

            printf("     加速度(g):  X=");
            print_g(ax_mg);
            printf("  Y=");
            print_g(ay_mg);
            printf("  Z=");
            print_g(az_mg);
            printf("   温度(℃): ");
            print_c(tp_100);
            printf("\r\n");

            printf("     角速度(dps): X=");
            print_dps(gx_100);
            printf("  Y=");
            print_dps(gy_100);
            printf("  Z=");
            print_dps(gz_100);
            printf("\r\n\r\n");

#endif
            reco_push(ax, ay, az, gx, gy, gz);
            reco_step();

#if RECO_FALL_ALARM_ENABLE
            if (s_tick_ms < s_alarm_until) {
                BEEP_ON();
            } else {
                BEEP_OFF();
            }
#endif

            if ((cnt % MPU_LED_TOGGLE_SAMPLES) == 0U) {
                LED_DS0_TOGGLE();
            }
            cnt++;
        } else {
            err_cnt++;
            printf("ERR,%lu,%lu\r\n", (unsigned long)cnt, (unsigned long)err_cnt);
            LED_DS0_OFF();
        }

        while ((int32_t)(next_ms - s_tick_ms) > 0) {
        }
    }
}
