#include <stdio.h>
#include "aspen.h"

char* time_info = BUILD_INFO_TIME;
char* gcc_info = BUILD_INFO_GCC;
char* uname_info = BUILD_INFO_UNAME;
char* branch_info = BUILD_INFO_BRANCH;
char* nvcc_info = BUILD_INFO_NVCC;
char* gpu_arch_info = BUILD_INFO_GPU_ARCH;
char* flag_info = BUILD_INFO_FLAGS;

void print_aspen_build_info(void)
{
    printf ("\n////////////    PRINTING ASPEN BUILD INFO    ////////////\n\n");
    printf ("1. Time of Build:\t%s\n", time_info);
    printf ("2. Build Commit:\t%s\n", branch_info);
    printf ("3. Build System:\t%s\n", uname_info);
    printf ("4. GCC Version:\t\t%s\n", gcc_info);
    printf ("5. NVCC Version:\t%s\n", nvcc_info);
    printf ("6. GPU ARCH:\t\t%s\n", gpu_arch_info);
    printf ("7. Build Flags:\t\t%s\n", flag_info);
    printf ("\n////////////////////////////////////////////////////////\n\n");
}