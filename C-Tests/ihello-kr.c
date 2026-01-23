#include <stdio.h>

int main() {
    char s[100];

    printf("Input 'hello' in Korean. It will echo your input.\n");
    printf("입력(echo): ");
    scanf_s("%s", s, (unsigned int)sizeof(s));
    printf("\n");
    
    printf("출력: %s\n", s);

    return 0;
}