#include <stdio.h>

int main(void) {
    char s[100];

    // 1. 영어 출력 테스트
    printf("==================================================\n");
    printf("[ENG] Hello, World!\n");
    printf("==================================================\n\n");

    // 2. 한국어 출력 테스트
    printf("--------------------------------------------------\n");
    printf("[KR] 안녕, 세계야!\n");
    printf("--------------------------------------------------\n\n");

    // 3. 인코딩 관련 영문 경고 문구
    printf("*****************************************************\n");
    printf("* NOTICE: IF KOREAN CHARACTERS ARE BROKEN           *\n");
    printf("* 1. Click 'Reopen with Encoding' (Bottom Right)    *\n");
    printf("* 2. Select 'EUC-KR'                                *\n");
    printf("* 3. If text corrupts, press 'Ctrl+Z' to undo       *\n");
    printf("* 4. Save the file(or press 'Ctrl+S') and run again *\n");
    printf("*****************************************************\n\n");

    // 4. 영어-한국어 입출력 테스트
    printf("Input 'hello, 세계야!' (Mixed Echo): ");
    scanf_s("%s", s, (unsigned int)sizeof(s));
    
    printf("\n[Echo Result]\n");
    printf("입력값 출력: %s\n", s);
    printf("==================================================\n");

    return 0;
}
