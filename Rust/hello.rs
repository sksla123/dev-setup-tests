use std::io::{self, Write};

fn main() {
    // 1. 영어 출력 테스트
    println!("==================================================");
    println!("[ENG] Hello, World!");
    println!("==================================================");
    println!();

    // 2. 한국어 출력 테스트
    println!("--------------------------------------------------");
    println!("[KR] 안녕, 세계야!");
    println!("--------------------------------------------------");
    println!();

    // 3. 영어-한국어 입출력 테스트 (Echo)
    println!("Multi-language input-output test(echo)");
    print!("Input 'hello, 세계야!' : ");
    
    // 입력을 받기 위해 stdout을 즉시 비움 (flush)
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    // 입력받은 문자열의 끝에 있는 개행 문자 제거
    let input = input.trim();

    println!("\n[Echo Result]");
    println!("입력값 출력: {}", input);
    println!("==================================================");
}