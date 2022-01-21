/**
*CPU Fibonacci code 
*/
#include <iostream>
#include <ctime>

using namespace std;


void Fibonacci(int n) {

	if (n == 0) {
		return 0;
	}
	if (n == 1) {
		return 1;
	}
	return Fibonacci(n - 1) + Fibonacci(n - 2);
}

int main() {

	auto begining = chrono::steady_clock::now();

	Fibonacci(47);

	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed_seconds = end - begining;
	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	

}