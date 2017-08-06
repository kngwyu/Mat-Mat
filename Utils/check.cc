#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <map>
using namespace std;
int main() {
    cout << "correct file: " << endl;
    map<pair<int, int>, double> mp;
    {
        string s;
        cin >> s;
        ifstream ifs(s);
        int i, j;
        double d;
        while (ifs >> i >> j >> d) 
            mp[make_pair(i, j)] = d;
    }
    cout << "check file: " << endl;
    string s;
    cin >> s;
    ifstream ifs(s);
    int i, j;
    double d;
    while (ifs >> i >> j >> d) {
        cerr << i << ' ' << j << ' ' << d << ' ' << mp[make_pair(i, j)] << endl;
        assert(fabs(mp[make_pair(i, j)] - d) < 1.0e-6);
    }
}
