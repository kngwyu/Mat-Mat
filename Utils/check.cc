#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <map>
using namespace std;
int main() {
    map<pair<int, int>, double> mp;
    {
        ifstream ifs("correct.dat");
        int i, j;
        double d;
        while (ifs >> i >> j >> d) 
            mp[make_pair(i, j)] = d;
    }
    ifstream ifs("check.dat");
    int i, j;
    double d;
    while (ifs >> i >> j >> d) {
        cout << d << endl;
        if(fabs(mp[make_pair(i, j)] - d) > 1.0e-6)
            cout << i << ' ' << j << ' ' << d << ' ' << mp[make_pair(i, j)] << endl;
    }
}
