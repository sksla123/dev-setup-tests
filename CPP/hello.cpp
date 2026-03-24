#include <iostream>
#include <string>

int main() {
    std::string s;

    // 1. 영어 출력 테스트
    std::cout << "==================================================" << std::endl;
    std::cout << "[ENG] Hello, World!" << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;

    // 2. 한국어 출력 테스트
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "[KR] 안녕, 세계야!" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl << std::endl;

    // 3. 인코딩 관련 영문 경고 문구
    std::cout << "*****************************************************" << std::endl;
    std::cout << "* NOTICE: IF KOREAN CHARACTERS ARE BROKEN           *" << std::endl;
    std::cout << "* 1. Click 'Reopen with Encoding' (Bottom Right)    *" << std::endl;
    std::cout << "* 2. Select 'EUC-KR'                                *" << std::endl;
    std::cout << "* 3. If text corrupts, press 'Ctrl+Z' to undo       *" << std::endl;
    std::cout << "* 4. Save the file(or press 'Ctrl+S') and run again *" << std::endl;
    std::cout << "*****************************************************" << std::endl << std::endl;

    // 4. 영어-한국어 입출력 테스트
    std::cout << "Input 'hello, 세계야!' (Mixed Echo): ";
    std::cin >> s;

    std::cout << "\n[Echo Result]" << std::endl;
    std::cout << "입력값 출력: " << s << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}