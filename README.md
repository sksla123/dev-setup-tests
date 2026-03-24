# Multi-Language Environment & Library Test

이 저장소는 다양한 프로그래밍 언어의 개발 환경 설치 여부를 확인하고, **한글/영문 혼용 입출력** 및 **텍스트 인코딩**이 정상적으로 작동하는지 테스트하기 위한 코드 모음을 포함하고 있습니다.

## 📋 테스트 목적
* 각 언어별 컴파일러 및 런타임 설치 확인
* 표준 입출력(Standard I/O)의 영문 및 한글 처리 능력 검증
* 개발 도구(VS Code 등)와 터미널 간의 인코딩 설정(UTF-8, EUC-KR) 호환성 체크
* 각 라이브러리 별 하드웨어 정상 인식 여부 확인

---

## 🚀 언어 별 초기 테스트 방법 (hello 파일)

### 1. C (`C\hello.c`)
Windows 환경의 CMD/PowerShell에서 한글이 깨질 경우를 대비한 안내 문구가 포함되어 있습니다.
* **컴파일:** `gcc hello.c -o hello.exe`
* **실행:** `./hello.exe`
* **주의:** 한글이 깨진다면 하단 Troubleshooting 파트를 확인하십시오.

### 2. C++ (`CPP\hello.cpp`)
C++ 표준 스트림(`iostream`)과 `std::string`을 이용한 테스트입니다.
* **컴파일:** `g++ hello.cpp -o hello_cpp.exe`
* **실행:** `./hello_cpp.exe`
* **주의:** 한글이 깨진다면 하단 Troubleshooting 파트를 확인하십시오.

### 3. Python (`Python\hello.py`)
Python 3의 기본 UTF-8 환경을 테스트합니다. 별도의 컴파일 과정 없이 즉시 실행 가능합니다.
* **실행:** `python -m hello`
* **UV를 통한 실행:** `uv run python -m hello`

### 4. Rust (`Rust\hello.rs`)
Rust의 강력한 UTF-8 지원과 `std::io` 라이브러리를 테스트합니다.
* **컴파일:** `rustc hello.rs`
* **실행:** `./hello.exe` (또는 `./hello`)
---

## 🛠 공통 테스트 시나리오
모든 소스 코드는 다음과 같은 순서로 테스트를 진행합니다:
1. **[ENG]** 기본적인 영문 문자열 출력 확인
2. **[KR]** 고정된 한글 문자열 출력 확인 (인코딩 검증)
3. **[Input-Output]** 사용자로부터 영문/한글 혼용 문자열을 입력받아 그대로 출력(Echo)하는 기능 확인

---

## ⚠️ 문제 해결 (Troubleshooting)
* **한글 깨짐 현상 (C/C++):** Windows 터미널의 기본 코드페이지와 소스 파일의 인코딩이 맞지 않을 때 발생합니다. 파일 인코딩을 `EUC-KR`로 변경하거나, 소스 코드 최상단에 로캘 설정을 추가해야 할 수 있습니다. 이 브랜치의 소스코드는 기본적으로 UTF-8 인코딩을 지향합니다.
    + 소스 코드 자체가 깨져있는 경우
        - VS Code 하단에서 UTF-8 클릭 -> Reopen with Encoding -> 저장
    + 터미널에서 입출력이 깨져있는 경우
        - VS Code 하단에서 UTF-8 클릭 -> Reopen with Encoding -> EUC-KR 선택 후, 글자가 깨지면 Ctrl+Z로 복구하여 저장하세요.
