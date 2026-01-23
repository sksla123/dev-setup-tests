#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;

    cout << "Input 'hello' in English. It will echo your input." << endl;
    cout << "Input(echo): ";
    cin >> s;
    cout << endl;
    
    cout << "Output: " << s << endl;

    return 0;
}