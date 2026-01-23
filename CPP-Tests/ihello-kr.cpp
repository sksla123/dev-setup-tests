#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;

    cout << "Input 'hello' in Korean. It will echo your input." << endl;
    cout << "입력(echo): ";
    cin >> s;
    cout << endl;
    
    cout << "출력: " << s << endl;

    return 0;
}