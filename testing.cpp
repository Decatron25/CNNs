#include<bits/stdc++.h>
using namespace std;

vector<vector<vector<vector<double>>>> inputData;
vector<vector<double>> trueOutputData;

void getData(string filename) {
    vector<double> row;
    string line, word;
    int rowCount = 0;
    fstream file (filename, ios::in);
    if(file.is_open())
    {
        while(getline(file, line))
        {


            rowCount++;
            if (rowCount == 1) {
                continue;
            }
            row.clear();
                
            stringstream str(line);



            int lastColVal;
            while(getline(str, word, ',')) {
                lastColVal = stoi(word);
                row.push_back(stod(word));
            }

            vector<double> trueOutput(10, 0);
            trueOutput[lastColVal] = 1.0;
            trueOutputData.push_back(trueOutput);

            if (row.size() > 0)
                row.pop_back();
            //cout<<rowCount<<" "<<row[129]<<endl;
            int dim = 28;
            vector<vector<vector<double>>> inputImage (1, vector<vector<double>> (dim, vector<double> (dim, 0)));
            for (int i = 0; i < row.size(); i++) {
                int r = i/dim;
                int c = i%dim;
                inputImage[0][r][c] = row[i];
            }
            

            inputData.push_back(inputImage);

        }
    }
}

void printData () {
    // for (int i = 0; i < inputData.size(); i++) {
    //     for (int j = 0; j < inputData[i].size(); j++) {
    //         for (int k = 0; k < inputData[i][j].size(); k++) {
    //             for (int l = 0; l < inputData[i][j][k].size(); l++) {
    //                 cout<<inputData[i][j][k][l]<<"";
    //             }
    //             cout<<"\n";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";
    // }

    cout<<inputData.size()<<" "<<inputData[0].size()<<" "<<inputData[0][0].size()<<" "<<inputData[0][0][0].size();
}

int main () {
    getData("train.csv");
    printData();
}