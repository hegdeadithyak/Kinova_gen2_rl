#include <bits/stdc++.h>
using namespace std;

#define fastio ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
using ll = long long;
using pii = pair<int,int>;
using vi = vector<int>;
using vl = vector<long long>;
const int MOD = 1e9+7;
const ll INF = 1e18;
#include <bits/stdc++.h>
using namespace std;

int findMaximumXOR(vector<int>& nums) {
	int max_val = 0;
	int mask = 0;
	for (int i = 30; i >= 0; i--) {
		mask = mask | (1 << i);
		
		unordered_set<int> prefixes;
		for (int num : nums) {
			prefixes.insert(num & mask);
		}
		
		int candidate = max_val | (1 << i);
		bool found = false;
		
		for (int p : prefixes) {
			if (prefixes.count(candidate ^ p)) {
				found = true;
				break;
			}
		}
		
		if (found) {
			max_val = candidate;
		}
	}
	return max_val;
}
void solve(){
	// Step 1: Read input

	// Step 2: Think about problem
	// Write your approach here before coding
	int n;
	cin >> n;
	vi a(n);
	
	for(auto &it:a)
	{
		cin >> it;
	}
	// Step 3: Implement solution
	cout << findMaximumXOR(a) << endl;
}

int main(){
	fastio;
	int t=1;
	cin >> t;
	while(t--) solve();
	return 0;
}