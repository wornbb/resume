#include <algorithm>
#include <array>
#include <bits/stdc++.h>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace std;
struct mtxCal{
    const int magic = 15;
    void insert(array<array<int, 3>, 3>& mtx, int loc, int val){
        int row = 2 - ((loc - 1) / 3);
        int col = loc % 3;
        mtx[row][col] = 3;
    }
    int row_sum(array<array<int, 3>, 3> mtx, int row){
        int result = 0;
        for (auto&& num : mtx[row]){
            result += num;
        }
        return result;
    }
    int col_sum(array<array<int, 3>, 3> mtx, int col){
        int result = 0;
        for (auto&& row : mtx){
            result += row[col];
        }
        return result;
    }
    bool isMagic(array<array<int, 3>, 3> mtx){
        for(int i = 0; i < 3; i++){
            if(row_sum(mtx, i) != magic){
                return false;
            }
        }
        for(int i = 0; i < 3; i++){
            if(col_sum(mtx, i) != magic){
                return false;
            }
        }
        return true;
    }
    int diff(vector<vector<int>> mtx1, array<array<int, 3>, 3> mtx2){
        int cost = 0;
        for(int row = 0; row < 3; row++){
            for(int col = 0; col < 3; col++){
                if (int c = mtx1[row][col] - mtx2[row][col]){
                    cost += c;
                }
            }
        }
        return cost;
    }
};
struct allMagic{
    mtxCal cal;
    array<array<int, 3>, 3> tmp_mtx;
    vector<array<array<int, 3>, 3>> magic_matrixs;
    allMagic(){
        vector<int> perm(9);
        iota(perm.begin(), perm.end(), 1);
    }
    void testAll(vector<int> perm){
        int s = perm.size();
        if (s == 0) {
            if(cal.isMagic(tmp_mtx)){
                magic_matrixs.push_back(tmp_mtx);
            }
        }else {
            for(int i = 0; i < s; i++){
                cal.insert(tmp_mtx, s, perm[i]);
                vector<int> tmp_perm = perm;
                tmp_perm.erase(tmp_perm.begin() + i);
                testAll(tmp_perm);
            }
        }
    }
};
// Complete the formingMagicSquare function below.
int formingMagicSquare(vector<vector<int>> s) {
    allMagic magics;
    vector<int> costs;
    for(auto&& mtx : magics.magic_matrixs){
        costs.push_back(magics.cal.diff(s, mtx));
    }
    return *min_element(costs.begin(), costs.end());
}

int main()
{
    allMagic magics;
    cout << magics.magic_matrixs.size();
    return 0;
}
