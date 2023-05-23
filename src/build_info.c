#include <stdio.h>
#include "aspen.h"

// char* time_info = BUILD_INFO_TIME;
char* gcc_info = BUILD_INFO_GCC;
char* uname_info = BUILD_INFO_UNAME;
char* branch_info = BUILD_INFO_BRANCH;
char* flag_info = BUILD_INFO_FLAGS;

void print_aspen_build_info(void)
{
    printf ("\n////////////    PRINTING ASPEN BUILD INFO    ////////////\n\n");
    printf ("1. Build Commit:\t%s\n", branch_info);
    printf ("2. Build System:\t%s\n", uname_info);
    printf ("3. GCC Version:\t\t%s\n", gcc_info);
    printf ("4. Build Flags:\t\t%s\n", flag_info);
    printf ("\n////////////////////////////////////////////////////////\n\n");
}