#include <stdio.h>

int main() {
    char s[100];

    printf("Input 'hello' in English. It will echo your input.\n");
    printf("Input(echo): ");
    scanf_s("%s", s, (unsigned int)sizeof(s));
    printf("\n");

    printf("Output: %s\n", s);

    return 0;
}