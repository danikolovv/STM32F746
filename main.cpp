#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h"  
#include "test_data.h"   
// CYCLE COUNT
#define DWT_CONTROL  (*((volatile uint32_t*)0xE0001000))
#define DWT_CYCCNT   (*((volatile uint32_t*)0xE0001004))
#define DEMCR        (*((volatile uint32_t*)0xE000EDFC))
// CACHE
#define SCB_ICIALLU  (*((volatile uint32_t*)0xE000EF50)) // Instruction cache invalidate
#define SCB_CCR      (*((volatile uint32_t*)0xE000ED14)) // Configuration and Control Register
#define SCB_CSSELR   (*((volatile uint32_t*)0xE000ED84)) // Cache Size Selection Register
#define SCB_DCISW    (*((volatile uint32_t*)0xE000EF60)) // Data cache invalidate by set-way
// --- SYSTEM STUBS ---
extern "C" {
    extern uint32_t _estack;
    extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss;
    void reset_handler(void); 
    int main();

    // Fix the USART1 Data Register address for STM32F7
    void DebugLog(const char* s) {
        volatile uint32_t *uart_tdr = (uint32_t *)0x40011028; 
        while (*s) {
            *uart_tdr = (uint32_t)(*s++);
        }
    }

    int DebugVsnprintf(char* buffer, size_t size, const char* format, va_list args) {
        return vsnprintf(buffer, size, format, args);; 
    }
}

// --- VECTOR TABLE ---
extern "C" {
    __attribute__((section(".isr_vector"), used))
    uint32_t * const g_pfnVectors[] = {
        (uint32_t *)&_estack,
        (uint32_t *)reset_handler,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
    };
}

// --- TFLITE GLOBALS ---
// Move these outside and use 'static' to keep them out of the stack
const int tensor_arena_size = 320 * 1024; 
alignas(16) static uint8_t tensor_arena[tensor_arena_size];

// --- RESET HANDLER ---
extern "C" void __attribute__((naked, used, section(".text.reset_handler"))) reset_handler(void) {
    asm volatile (
        "ldr r0, =0xE000ED88          \n" // CPACR
        "ldr r1, [r0]                 \n"
        "orr r1, r1, #(0xF << 20)     \n" // Enable CP10 and CP11
        "str r1, [r0]                 \n"
        "dsb sy                       \n"
        "isb sy                       \n"
        // arm-none-eabi-objdump -d cifar_eval_m7.elf | grep -E "vadd|vsub|vmul|vldr" | head -n 20
        "ldr r3, =_estack          \n"
        "msr msp, r3               \n"
        "isb                       \n"
        "ldr r0, =_sdata           \n"
        "ldr r1, =_sidata          \n"
        "ldr r2, =_edata           \n"
        "copy_loop:                \n"
        "cmp r0, r2                \n"
        "itt lt                    \n"
        "ldrlt r3, [r1], #4        \n"
        "strlt r3, [r0], #4        \n"
        "blt copy_loop             \n"
        "ldr r0, =_sbss            \n"
        "ldr r1, =_ebss            \n"
        "mov r2, #0                \n"
        "zero_loop:                \n"
        "cmp r0, r1                \n"
        "it lt                     \n"
        "strlt r2, [r0], #4        \n"
        "blt zero_loop             \n"
        "ldr r0, =main             \n"
        "orr r0, r0, #1            \n" 
        "bx r0                     \n"
        ".pool                     \n" 
    );
}

void my_itoa(int n, char s[], int base) {
    int i = 0;
    // Handle 0 explicitly
    if (n == 0) {
        s[i++] = '0';
        s[i] = '\0';
        return;
    }
    // Convert digits to characters (backwards)
    while (n > 0) {
        s[i++] = (n % 10) + '0';
        n /= 10;
    }
    s[i] = '\0';
    // Reverse the string
    for (int j = 0, k = i - 1; j < k; j++, k--) {
        char temp = s[j];
        s[j] = s[k];
        s[k] = temp;
    }
}

void PrintFloat(float f, int precision = 3) {
    if (f < 0) {
        DebugLog("-");
        f = -f;
    }

    // 1. Print the integer part
    int int_part = (int)f;
    char int_buf[12];
    my_itoa(int_part, int_buf, 10);
    DebugLog(int_buf);
    DebugLog(".");

    // 2. Print the fractional part
    float fractional = f - (float)int_part;
    for (int i = 0; i < precision; i++) {
        fractional *= 10;
        int digit = (int)fractional;
        char digit_buf[2];
        my_itoa(digit, digit_buf, 10);
        DebugLog(digit_buf);
        fractional -= digit;
    }
}

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#define SCB_VTOR (*((volatile uint32_t *)0xE000ED08))

void PrintCifarLabel(int idx) {
    switch(idx) {
        case 0: DebugLog("airplane"); break;
        case 1: DebugLog("automobile"); break;
        case 2: DebugLog("bird"); break;
        case 3: DebugLog("cat"); break;
        case 4: DebugLog("deer"); break;
        case 5: DebugLog("dog"); break;
        case 6: DebugLog("frog"); break;
        case 7: DebugLog("horse"); break;
        case 8: DebugLog("ship"); break;
        case 9: DebugLog("truck"); break;
        default: DebugLog("unknown"); break;
    }
}
void Enable_ICache() {
    SCB_ICIALLU = 0;             // Invalidate I-Cache
    __asm volatile ("dsb sy");   // Data Synchronization Barrier
    __asm volatile ("isb sy");   // Instruction Synchronization Barrier
    SCB_CCR |= (1UL << 17);      // Set IC bit in CCR (Bit 17)
    __asm volatile ("dsb sy");
    __asm volatile ("isb sy");
}

void Enable_DCache() {
    // Note: In a real chip, you'd invalidate the D-cache by set/way here
    SCB_CCR |= (1UL << 16);      // Set DC bit in CCR (Bit 16)
    __asm volatile ("dsb sy");
    __asm volatile ("isb sy");
}

// --- MAIN ---
int main() {

    Enable_ICache();
    Enable_DCache();

    SCB_VTOR = 0x08000000;
    // 1. Hardware Init (Enable USART1)
    *(volatile uint32_t*)(0x40023830) |= (1 << 0); // GPIOA Clock
    *(volatile uint32_t*)(0x40023844) |= (1 << 4); // USART1 Clock
    *(volatile uint32_t*)(0x40011000) |= (1 << 0) | (1 << 3); // UE and TE

    // 1.2 Error Handler
    *((volatile uint32_t *)0xE000ED14) &= ~(1 << 4);

    // 1.3 FPU
    *(volatile uint32_t*)(0xE000ED88) |= ((3UL << 10*2) | (3UL << 11*2));

    __asm volatile ("dsb sy");
    __asm volatile ("isb sy");

    DEMCR |= (1 << 24);     // Set TRCENA
    DWT_CYCCNT = 0;         // Reset counter
    DWT_CONTROL |= (1 << 0); // Enable CYCCNT

    // 2. TFLite Setup
    const tflite::Model* model = tflite::GetModel(cifar10_model_tflite);
    
    // INCREASED TO 20 CAPACITY
    static tflite::MicroMutableOpResolver<30> resolver;

    // 2. Add each op and check the result ONCE
    if (resolver.AddConv2D() != kTfLiteOk)         DebugLog("Fail: Conv2D\n");
    if (resolver.AddMaxPool2D() != kTfLiteOk)      DebugLog("Fail: MaxPool\n");
    if (resolver.AddReshape() != kTfLiteOk)        DebugLog("Fail: Reshape\n");
    if (resolver.AddFullyConnected() != kTfLiteOk) DebugLog("Fail: FullyConnected\n");
    if (resolver.AddRelu() != kTfLiteOk)           DebugLog("Fail: Relu\n");
    if (resolver.AddSoftmax() != kTfLiteOk)        DebugLog("Fail: Softmax\n");
    if (resolver.AddMul() != kTfLiteOk)            DebugLog("Fail: Mul\n");
    if (resolver.AddAdd() != kTfLiteOk)            DebugLog("Fail: Add\n");
    if (resolver.AddSqueeze() != kTfLiteOk)        DebugLog("Fail: Squeeze\n");
    if (resolver.AddQuantize() != kTfLiteOk)       DebugLog("Fail: Quantize\n");
    if (resolver.AddShape() != kTfLiteOk)       DebugLog("Fail: Shape\n");
    if (resolver.AddStridedSlice() != kTfLiteOk)       DebugLog("Fail: StridedSlice\n");
    if (resolver.AddPack() != kTfLiteOk)       DebugLog("Fail: Pack\n");
    // ... add any others you need ...

    // 3. Initialize the interpreter AFTER all ops are added
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, tensor_arena_size, nullptr);

    TfLiteStatus allocate_status = interpreter.AllocateTensors();

    if (allocate_status != kTfLiteOk) {
        if (allocate_status == 1) DebugLog("Status is kTfLiteError\n");
        if (allocate_status == 5) DebugLog("Status is kTfLiteUnresolvedOps\n");
    }

    if (allocate_status != kTfLiteOk) {
        // %d tells MicroPrintf to treat 'allocate_status' as an integer
        MicroPrintf("Allocation failed! Error code: %d", (int)allocate_status);
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    DWT_CYCCNT = 0;
    // 4. Inference Loop
    for (int j = 0; j < 10; ++j) {
        uint32_t start = DWT_CYCCNT;
        
        memcpy(input->data.f, image_sample_0[j], 32 * 32 * 3 * sizeof(float));

        if (interpreter.Invoke() != kTfLiteOk) {
            DebugLog("Error: Invoke failed!\n");
            continue;
        }

        // Output the first class probability to the UART for proof of life
        float top_result = output->data.f[0];
        if (top_result > 0.5f) {
            DebugLog("High Confidence Prediction Detected!\n");
        }

        float max_val = -1.0f;
        int max_idx = -1;
        for (int i = 0; i < 10; ++i) {
            if (output->data.f[i] > max_val) {
                max_val = output->data.f[i];
                max_idx = i;
            }
        }

        DebugLog("Sample ");
        char sample_buf[4];
        my_itoa(j, sample_buf, 10);
        DebugLog(sample_buf);

        DebugLog(": Class ");
        char class_buf[4];
        my_itoa(max_idx, class_buf, 10);
        DebugLog(class_buf);

        DebugLog(", Label: ");
        PrintCifarLabel(max_idx);

        DebugLog(" Confidence: ");
        PrintFloat(max_val, 4);
        DebugLog("\n");
        DebugLog("Correct Class: ");
        char real_class_id[4];
        my_itoa(test_labels[j], real_class_id, 10);
        DebugLog(real_class_id); // Prints "3"

        DebugLog(", Label: ");
        // Use the integer test_labels[j] directly to index the array
        PrintCifarLabel(test_labels[j]);
        DebugLog("\n");
    }

    DebugLog("All Inferences Complete. Model is stable.\n");
    MicroPrintf("BENCHMARK_COMPLETE\n");

    uint32_t total_cycles = DWT_CYCCNT;
    while(1); 
}
