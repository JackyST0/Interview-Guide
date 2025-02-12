# LeetCode 热题 100

## 1. 两数之和
```
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target  的那两个整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2：
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3：
输入：nums = [3,3], target = 6
输出：[0,1]
```
- 题解：  
    - 暴力枚举
        ```
        class Solution {
            public int[] twoSum(int[] nums, int target) {
                    int n = nums.length;
                    for (int i = 0; i < n; ++i) {
                        for (int j = i + 1; j < n; ++j) {
                            if (nums[i] + nums[j] == target) {
                                return new int[]{i,j};
                            }
                        }
                    }
                    return new int[0];
            }
        }
        ```
        - 时间复杂度：O(N²)  
        - 空间复杂度：O(1)
    -  哈希表
        ```
        class Solution {
            public int[] twoSum(int[] nums, int target) {
                Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
                for (int i = 0; i < nums.length; ++i) {
                    if (hashtable.containsKey(target - nums[i])) {
                        return new int[]{hashtable.get(target - nums[i]), i};
                    }
                    hashtable.put(nums[i], i);
                }
                return new int[0];
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)

## 2. 字母异位词分组
```
给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。

字母异位词是由重新排列源单词的所有字母得到的一个新单词。

示例 1:
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

示例 2:
输入: strs = [""]
输出: [[""]]

示例 3:
输入: strs = ["a"]
输出: [["a"]]
```
- 题解
    - 排序
        ```
        class Solution {
            public List<List<String>> groupAnagrams(String[] strs) {
                Map<String, List<String>> map = new HashMap<String, List<String>>();
                for (String str : strs) {
                    char[] array = str.toCharArray();
                    Arrays.sort(array);
                    String key = new String(array);
                    List<String> list = map.getOrDefault(key, new ArrayList<String>());
                    list.add(str);
                    map.put(key, list);
                }
                return new ArrayList<List<String>>(map.values());
            }
        }
        ```
        - 时间复杂度：O(nklogk)  
        - 空间复杂度：O(nk)

## 3. 最长连续序列
```
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

示例 2：
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```
- 题解
    - 哈希表
        ```
        class Solution {
            public int longestConsecutive(int[] nums) {
                Set<Integer> num_set = new HashSet<Integer>();
                for (int num : nums) {
                    num_set.add(num);
                }

                int longestStreak = 0;

                for (int num : num_set) {
                    if (!num_set.contains(num - 1)) {
                        int currentNum = num;
                        int currentStreak = 1;

                        while (num_set.contains(currentNum + 1)) {
                            currentNum += 1;
                            currentStreak += 1;
                        }

                        longestStreak = Math.max(longestStreak, currentStreak);
                    }
                }
                return longestStreak;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    
## 4. 移动零
```
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。

示例 1:
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]

示例 2:
输入: nums = [0]
输出: [0]
```
- 题解
    - 两次遍历
        ```
        class Solution {
            public void moveZeroes(int[] nums) {
                if (nums == null) {
                    return;
                }
                // 第一次遍历的时候，j指针记录非0的个数，只要是非0的通通都赋给nums[j]
                int j = 0;
                for (int i = 0; i < nums.length; ++i) {
                    if (nums[i] != 0) {
                        // 这里是先赋值之后，j才等于j+1
                        nums[j++] = nums[i];
                    }
                }
                // 非0元素统计完了，剩下的就是0了
                // 所以第二次遍历把末尾的元素都赋为0即可
                for (int i = j; i < nums.length; ++i) {
                    nums[i] = 0;
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)
    - 一次遍历
        ```
        class Solution {
            public void moveZeroes(int[] nums) {
                if (nums == null) {
                    return;
                }
                // 两个指针i和j
                int j = 0;
                for (int i = 0; i < nums.length; i++) {
                    // 当前元素!=0，就把其交换到左边，等于0的交换到右边
                    if (nums[i] != 0) {
                        int tmp = nums[i];
                        nums[i] = nums[j];
                        // 这里是先赋值之后，j才等于j+1
                        nums[j++] = tmp;
                    }
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 5. 盛最多水的容器
```
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。

示例 1：
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

示例 2：
输入：height = [1,1]
输出：1
```
![盛最多水的容器-示例1](/相关图册/盛最多水的容器-示例1.jpg)
- 题解
    - 双指针
        ```
        class Solution {
            public int maxArea(int[] height) {
                int l = 0, r = height.length - 1;
                int ans = 0;
                while (l < r) {
                    int area = Math.min(height[l], height[r]) * (r - l);
                    ans = Math.max(ans, area);
                    if (height[l] <= height[r]) {
                        ++l;
                    }
                    else {
                        --r;
                    }
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)

## 6. 三数之和
```
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

示例 1：
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。

示例 2：
输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。

示例 3：
输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。
```
- 题解
    - 排序 + 双指针
        ```
        class Solution {
            public List<List<Integer>> threeSum(int[] nums) {
                List<List<Integer>> ans = new ArrayList();
                int len = nums.length;
                if (nums == null || len < 3) {
                    return ans;
                }
                // 排序
                Arrays.sort(nums);
                for (int i = 0; i < len; i++) {
                    // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
                    if (nums[i] > 0) {
                        break;
                    }
                    // 去重
                    if (i > 0 && nums[i] == nums[i-1]) {
                        continue;
                    }
                    int L = i + 1;
                    int R = len - 1;
                    while(L < R) {
                        int sum = nums[i] + nums[L] + nums[R];
                        if (sum == 0) {
                            ans.add(Arrays.asList(nums[i], nums[L],nums[R]));
                            // 去重
                            while (L < R && nums[L] == nums[L + 1]) {
                                L++;
                            }
                            // 去重
                            while (L < R && nums[R] == nums[R-1]) {
                                R--;
                            }
                            L++;
                            R--;
                        }
                        else if (sum < 0) {
                            L++;
                        }
                        else if (sum > 0) {
                            R--;
                        }
                    }
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n²)  
        - 空间复杂度：O(n)

## 7. 接雨水
```
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

示例 1：
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

示例 2：
输入：height = [4,2,0,3,2,5]
输出：9
```
![接雨水-示例1](/相关图册/接雨水-示例1.png)
- 题解
    - 按列求
        ```
        class Solution {
            public int trap(int[] height) {
                int sum = 0;
                // 最两端的列不用考虑，因为一定不会有水。所以下标从 1 到 length - 2
                for (int i = 1; i < height.length - 1; i++) {
                    int max_left = 0;
                    // 找出左边最高
                    for (int j = i - 1; j >= 0; j--) {
                        if (height[j] > max_left) {
                            max_left = height[j];
                        }
                    }
                    int max_right = 0;
                    // 找出右边最高;
                    for (int j = i + 1;j < height.length; j++) {
                        if (height[j] > max_right) {
                            max_right = height[j];
                        }
                    }
                    // 找出两端较小的
                    int min = Math.min(max_left, max_right);
                    // 只有较小的一段大于当前列的高度才会有水，其他情况不会有水
                    if (min > height[i]) {
                        sum = sum + (min - height[i]);
                    }
                }
            return sum;
            }
        }
        ```
        - 时间复杂度：O(n²)  
        - 空间复杂度：O(1)
    - 双指针
        ```
        class Solution {
            public int trap(int[] height) {
                int sum = 0;
                int max_left = 0;
                int max_right = 0;
                int left = 1;
                // 加右指针进去
                int right = height.length - 2;
                for (int i = 1; i < height.length - 1; i++) {
                    // 从左到右
                    if (height[left - 1] < height[right + 1]) {
                        max_left = Math.max(max_left, height[left - 1]);
                        int min = max_left;
                        if (min > height[left]) {
                            sum = sum + (min - height[left]);
                        }
                        left++;
                        // 从右到左更
                    } else {
                        max_right = Math.max(max_right, height[right + 1]);
                        int min = max_right;
                        if (min > height[right]) {
                            sum = sum + (min - height[right]);
                        }
                        right--;
                    }
                }
            return sum;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 8. 无重复字符的最长子串
```
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

示例 2:
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

示例 3:
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```
- 题解
    - 活动窗口
        ```
        class Solution {
            public int lengthOfLongestSubstring(String s) {
                // 哈希集合，记录每个字符是否出现过
                Set<Character> occ = new HashSet<Character>();
                int n = s.length();
                // 右指针，初始值为 -1 ，相当于我们在字符串的左边界的左侧，还没有开始移动
                int rk = -1, ans = 0;
                for (int i = 0; i < n; ++i) {
                    if (i != 0) {
                        // 左指针向右移动一格，移除一个字符
                        occ.remove(s.charAt(i - 1));
                    }
                    while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))) {
                        // 不断的移动右指针
                        occ.add(s.charAt(rk + 1));
                        ++rk;
                    }
                    // 第 i 到 rk 个字符是一个极长的无重复字符子串
                    ans = Math.max(ans, rk -i + 1);
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)

## 9. 找到字符串中所有字母异位词
```
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

示例 1:
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

 示例 2:
输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
```
- 题解
    - 滑动窗口
        ```
        class Solution {
            public List<Integer> findAnagrams(String s, String p) {
                int sLen = s.length();
                int pLen = p.length();
                List<Integer> ans = new ArrayList<>();

                if (sLen < pLen) {
                    return ans;
                }
                // 建立两个数组存放字符串中字母出现的词频，并以此作为标准比较
                int[] sCount = new int[26];
                int[] pCount = new int[26];

                // 当滑动窗口的首位在s[0]处时（相当于防止滑动窗口进入数组）
                for (int i = 0; i < pLen; i++) {
                    // 记录s中前pLen个字母的词频
                    ++sCount[s.charAt(i) - 'a'];
                    // 记录要寻找的字符串中没个字母的词频（只用进行一次来确定）
                    ++pCount[p.charAt(i) - 'a'];
                }

                // 判断放置处是否有异位词（在放置时只需判断一次）
                if (Arrays.equals(sCount, pCount)) {
                    ans.add(0);
                }

                // 开始让窗口进行滑动
                // i是滑动前的首位
                for (int i = 0; i < sLen - pLen; i++) {
                    // 将滑动前首位的词频删去
                    --sCount[s.charAt(i) - 'a'];
                    // 增加滑动后最后一位的词频（以此达到滑动的效果）
                    ++sCount[s.charAt(i + pLen) - 'a'];

                    // 判断滑动后处，是否有异位词
                    if (Arrays.equals(sCount, pCount)) {
                        ans.add(i + 1);
                    } 
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 10. 和为 K 的子数组
```
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

子数组是数组中元素的连续非空序列。

示例 1：
输入：nums = [1,1,1], k = 2
输出：2

示例 2：
输入：nums = [1,2,3], k = 3
输出：2
```
- 题解
    - 枚举
        ```
        class Solution {
            public int subarraySum(int[] nums, int k) {
                int count = 0;
                for (int start = 0; start < nums.length; ++start) {
                    int sum = 0;
                    for (int end = start; end >= 0; --end) {
                        sum += nums[end];
                        if (sum == k) {
                            count++;
                        }
                    }
                }
                return count;
            }
        }
        ```
        - 时间复杂度：O(n²)  
        - 空间复杂度：O(1)
    - 前缀和
        ```
        class Solution {
            public int subarraySum(int[] nums, int k) {
                int len = nums.length;
                // 计算前缀和数组
                int[] preSum = new int[len + 1];
                preSum[0] = 0;
                for (int i = 0; i < len; i++) {
                    preSum[i + 1] = preSum[i] + nums[i];
                }

                int count = 0;
                for (int left = 0; left < len; left++) {
                    for (int right = left; right < len; right++) {
                        // 区间和 [left..right],注意下标偏移
                        if (preSum[right + 1] - preSum[left] == k) {
                            count++;
                        }
                    }
                }
                return count;
            }
        }
        ```
        - 时间复杂度：O(N²)  
        - 空间复杂度：O(N)

## 11. 滑动窗口最大值
```
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

示例 1：
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

示例 2：
输入：nums = [1], k = 1
输出：[1]
```
- 题解
    - 双向队列
        ```
        class Solution {
            public int[] maxSlidingWindow(int[] nums, int k) {
                if (nums == null || nums.length < 2) {
                    return nums;
                }
                // 双向队列 保存当前窗口最大值的数组位置 保证队列中数组位置的数值按从大到小排序
                LinkedList<Integer> queue = new LinkedList();
                // 结果数组
                int[] result = new int[nums.length - k + 1];
                // 遍历nums数组
                for (int i = 0; i <nums.length; i++) {
                    // 保证从大到小 如果前面数小则需要一次弹出，直至满足要求
                    while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
                        queue.pollLast();
                    }
                    // 添加当前值对应的数组下标
                    queue.addLast(i);
                    // 判断当前队列中队首的值是否有效
                    if (queue.peek() <= i - k) {
                        queue.poll();
                    }
                    // 当窗口长度为k时 保存当前窗口最大值
                    if (i + 1 >= k) {
                        result[i + 1 - k] = nums[queue.peek()];
                    }
                }
                return result;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(k)

## 12. 最小覆盖子串
```
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。
 
示例 1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。

示例 2：
输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。

示例 3:
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，因此没有符合条件的子字符串，返回空字符串。
```
- 题解
    - 滑动窗口
        ```
        class Solution {
            Map<Character, Integer> ori = new HashMap<Character, Integer>();
            Map<Character, Integer> cnt = new HashMap<Character, Integer>();

            public String minWindow(String s, String t) {
                int tLen = t.length();
                for (int i = 0; i < tLen; i++) {
                    char c =t.charAt(i);
                    ori.put(c, ori.getOrDefault(c, 0) + 1);
                }
                // l为左指针，r为右指针（逻辑位置）
                int l = 0, r = -1;
                // len为窗口长度，ansL为窗口左边界，ansR为窗口右边界（物理位置）
                int len = Integer.MAX_VALUE, ansL = -1, ansR = -1;
                int sLen = s.length();
                while (r < sLen) {
                    ++r;
                    if (r < sLen && ori.containsKey(s.charAt(r))) {
                        cnt.put(s.charAt(r), cnt.getOrDefault(s.charAt(r), 0) + 1);
                    }
                    while (check() && l <= r) {
                        if (r - l + 1 < len) {
                            len = r - l + 1;
                            ansL = l;
                            ansR =l + len;
                        }
                        if (ori.containsKey(s.charAt(l))) {
                            cnt.put(s.charAt(l), cnt.getOrDefault(s.charAt(l), 0) - 1);
                        }
                        ++l;
                    }
                }
                return ansL == -1 ? "" : s.substring(ansL, ansR);
            }
        
            public boolean check() {
                Iterator iter = ori.entrySet().iterator();
                while (iter.hasNext()) {
                    Map.Entry entry = (Map.Entry) iter.next();
                    Character key = (Character) entry.getKey();
                    Integer val = (Integer) entry.getValue();
                    if (cnt.getOrDefault(key, 0) < val) {
                        return false;
                    }
                }
                return true;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(k)

## 13. 最大子数组和
```
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组
是数组中的一个连续部分。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

示例 2：
输入：nums = [1]
输出：1

示例 3：
输入：nums = [5,4,-1,7,8]
输出：23
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int maxSubArray(int[] nums) {
                int pre = 0, maxAns = nums[0];
                for (int x : nums) {
                    pre = Math.max(pre + x, x);
                    maxAns = Math.max(maxAns, pre);
                }
                return maxAns;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 14. 合并区间
```
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

示例 1：
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2：
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```
- 题解
    - 排序
        ```
        class Solution {
            public int[][] merge(int[][] intervals) {
                if (intervals.length == 0) {
                    return new int[0][2];
                }
                Arrays.sort(intervals, new Comparator<int []>() {
                    public int compare(int[] interval1, int[] interval2) {
                        return interval1[0] - interval2[0];
                    }
                });
                List<int []> merged = new ArrayList<int []>();
                for (int i = 0 ; i < intervals.length ; ++i) {
                    int L = intervals[i][0], R = intervals[i][1];
                    if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                        merged.add(new int[]{L, R});
                    } else {
                        merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
                    }
                }
                return merged.toArray(new int[merged.size()][]);
            }
        }
        ```
        - 时间复杂度：O(nlogn)  
        - 空间复杂度：O(logn)

## 15. 轮转数组
```
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]
```
- 题解
    - 使用额外的数组
        ```
        class Solution {
            public void rotate(int[] nums, int k) {
                int n = nums.length;
                int[] newArr = new int[n];
                for (int i = 0 ; i < n ; ++i) {
                    newArr[(i + k) % n] = nums[i];
                }
                System.arraycopy(newArr, 0, nums, 0, n);
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 数组翻转
        ```
        class Solution {
            public void rotate(int[] nums, int k) {
                k %= nums.length;
                reverse(nums, 0, nums.length - 1);
                reverse(nums, 0, k - 1);
                reverse(nums, k, nums.length - 1);
            }

            public void reverse(int[] nums, int start, int end) {
                while (start < end) {
                    int temp = nums[start];
                    nums[start] = nums[end];
                    nums[end] = temp;
                    start += 1;
                    end -= 1;
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 16. 除自身以外数组的乘积
```
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请 不要使用除法，且在 O(n) 时间复杂度内完成此题。

示例 1:
输入: nums = [1,2,3,4]
输出: [24,12,8,6]

示例 2:
输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
```
- 题解
    - 左右乘积列表
        ```
        class Solution {
            public int[] productExceptSelf(int[] nums) {
                int length = nums.length;

                // L 和 R 分别表示左右两侧的成绩列表
                int[] L = new int[length];
                int[] R = new int[length];

                int[] answer = new int[length];

                // L[i] 为索引 i 左侧所有元素的乘积
                // 对于索引为 '0' 的元素，因为左侧没有元素。所以 L[0] = 1
                L[0] = 1;
                for (int i = 1; i < length; i++) {
                    L[i] = nums[i - 1] * L[i - 1];
                }

                // R[i] 为索引 i 右侧所有元素的乘积
                // 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
                R[length - 1] = 1;
                for (int i = length - 2; i >= 0; i--) {
                    R[i] = nums[i + 1] * R[i + 1];
                }

                // 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
                for (int i = 0; i < length; i++) {
                    answer[i] = L[i] * R[i];
                }

                return answer;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)
    - 空间复杂度 O(1) 的方法
        ```
        class Solution {
            public int[] productExceptSelf(int[] nums) {
                int length = nums.length;
                int[] answer = new int[length];

                // answer[i] 表示索引 i 左侧所有元素的乘积
                // 因为索引为'0' 的元素左侧没有元素，所以 answer[0] = 1
                answer[0] = 1;
                for (int i = 1; i < length; i++) {
                    answer[i] = nums[i - 1] * answer[i - 1];
                }

                // R 为右侧所有元素的乘积
                // 刚开始右边没有元素，所以 R = 1
                int R = 1;
                for (int i = length - 1; i >= 0; i--) {
                    // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
                    answer[i] = answer[i] * R;
                    // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
                    R *= nums[i];
                }
                return answer;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)

## 17. 缺失的第一个正数
```
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
 
示例 1：
输入：nums = [1,2,0]
输出：3
解释：范围 [1,2] 中的数字都在数组中。

示例 2：
输入：nums = [3,4,-1,1]
输出：2
解释：1 在数组中，但 2 没有。

示例 3：
输入：nums = [7,8,9,11,12]
输出：1
解释：最小的正数 1 没有出现。
```
- 题解
    - 哈希表
        ```
        class Solution {
            public int firstMissingPositive(int[] nums) {
                int n = nums.length;
                for (int i = 0; i < n; ++i) {
                    if (nums[i] <= 0) {
                        nums[i] = n + 1;
                    }
                }
                for (int i = 0; i < n; ++i) {
                    int num = Math.abs(nums[i]);
                    if (num <= n) {
                        nums[num - 1] = -Math.abs(nums[num - 1]);
                    }
                }
                for (int i = 0; i < n; ++i) {
                    if (nums[i] > 0) {
                        return i + 1;
                    }
                }
                return n + 1;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)
    - 置换
        ```
        class Solution {
            public int firstMissingPositive(int[] nums) {
                int n = nums.length;
                for (int i = 0; i < n; ++i) {
                    while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                        int temp = nums[nums[i] - 1];
                        nums[nums[i] - 1] = nums[i];
                        nums[i] = temp;
                    }
                }
                for (int i = 0; i < n; ++i) {
                    if (nums[i] != i + 1) {
                        return i + 1;
                    }
                }
                return n + 1;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)

## 18. 矩阵置零  
```
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

示例 1：
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]

示例 2：
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
``` 
![矩阵置零-示例1](/相关图册/矩阵置零-示例1.jpg)  
![矩阵置零-示例2](/相关图册/矩阵置零-示例2.jpg)
- 题解
    - 使用标记数组
        ```
        class Solution {
            public void setZeroes(int[][] matrix) {
                int m = matrix.length, n = matrix[0].length;
                boolean[] row = new boolean[m];
                boolean[] col = new boolean[n];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        if (matrix[i][j] == 0) {
                            row[i] = col[j] = true;
                        }
                    }
                }
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        if (row[i] || col[j]) {
                            matrix[i][j] = 0;
                        }
                    }
                }
            }
        }
        ```
        - 时间复杂度：O(mn)  
        - 空间复杂度：O(m+n)
    - 使用两个标记变量
        ```
        class Solution {
            public void setZeroes(int[][] matrix) {
                int m = matrix.length, n = matrix[0].length;
                boolean flagCol0 = false, flagRow0 = false;
                for (int i = 0; i < m; i++) {
                    if (matrix[i][0] == 0) {
                        flagCol0 = true;
                    }
                }
                for (int j = 0; j < n; j++) {
                    if (matrix[0][j] == 0) {
                        flagRow0 = true;
                    }
                }
                for (int i = 1; i < m; i++) {
                    for (int j = 1; j < n; j++) {
                        if (matrix[i][j] == 0) {
                            matrix[i][0] = matrix[0][j] = 0;
                        }
                    }
                }
                for (int i = 1; i < m; i++) {
                    for (int j = 1; j < n; j++) {
                        if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                            matrix[i][j] = 0;
                        }
                    }
                }
                if (flagCol0) {
                    for (int i = 0; i < m; i++) {
                        matrix[i][0] = 0;
                    }
                }
                if (flagRow0) {
                    for (int j = 0; j < n; j++) {
                        matrix[0][j] = 0;
                    }
                }
            }
        }
        ```
        - 时间复杂度：O(mn)  
        - 空间复杂度：O(1)
    
## 19. 螺旋矩阵
```
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

示例 1：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]

示例 2：
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```
![螺旋矩阵-示例1](/相关图册/螺旋矩阵-示例1.jpg)  
![螺旋矩阵-示例2](/相关图册/螺旋矩阵-示例2.jpg)
- 题解
    - 模拟
        ```
        class Solution {
            public List<Integer> spiralOrder(int[][] matrix) {
                List<Integer> order = new ArrayList<Integer>();
                if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
                    return order;
                }
                int rows = matrix.length, columns = matrix[0].length;
                boolean[][] visited = new boolean[rows][columns];
                int total = rows * columns;
                int row = 0, column = 0;
                int [][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
                int directionIndex = 0;
                for (int i = 0; i < total; i++) {
                    order.add(matrix[row][column]);
                    visited[row][column] = true;
                    int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
                    if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                        directionIndex = (directionIndex + 1) % 4;  
                    }
                    row += directions[directionIndex][0];
                    column += directions[directionIndex][1];
                }
                return order;
            }
        }
        ```
        - 时间复杂度：O(mn)  
        - 空间复杂度：O(mn)
    - 按层模拟
        ```
        class Solution {
            public List<Integer> spiralOrder(int[][] matrix) {
                List<Integer> order = new ArrayList<Integer>();
                if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
                    return order;
                }
                int rows = matrix.length, columns = matrix[0].length;
                int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
                while (left <= right && top <= bottom) {
                    for (int column = left; column <= right; column++) {
                        order.add(matrix[top][column]);
                    }
                    for (int row = top + 1; row <= bottom; row++) {
                        order.add(matrix[row][right]);
                    }
                    if (left < right && top < bottom) {
                        for (int column = right - 1; column > left; column--) {
                            order.add(matrix[bottom][column]);
                        }
                        for (int row = bottom; row > top; row--) {
                            order.add(matrix[row][left]);
                        }
                    }
                    left++;
                    right--;
                    top++;
                    bottom--;
                }
                return order;
            }
        }
        ```
        - 时间复杂度：O(mn)  
        - 空间复杂度：O(1)
    
## 20.旋转图像
```
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

示例 1：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]

示例 2：
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```
![旋转图像-示例1](/相关图册/旋转图像-示例1.jpg)    
![旋转图像-示例2](/相关图册/旋转图像-示例2.jpg)
- 题解 
    - 使用辅助数组
        ```
        class Solution {
            public void rotate(int[][] matrix) {
                int n = matrix.length;
                int[][] matrix_new = new int[n][n];
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        matrix_new[j][n - i -1] = matrix[i][j];
                    }
                }
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        matrix[i][j] = matrix_new[i][j];
                    }
                }
            }
        }
        ```
        - 时间复杂度：O(N²)  
        - 空间复杂度：O(N²)
    - 用翻转代替旋转
        ```
        class Solution {
            public void rotate(int[][] matrix) {
                int n = matrix.length;
                // 水平翻转
                for (int i = 0; i < n / 2; ++i) {
                    for (int j  = 0; j < n; ++j) {
                        int temp = matrix[i][j];
                        matrix[i][j] = matrix[n - i - 1][j];
                        matrix[n - i - 1][j] = temp;
                    }
                }
                // 主对角线翻转
                for (int i = 0; i < n; ++i) {
                    for ( int j = 0; j < i; ++j) {
                        int temp = matrix[i][j];
                        matrix[i][j] = matrix[j][i];
                        matrix[j][i] = temp;
                    }
                }    
            }
        }
        ```
        - 时间复杂度：O(N²)  
        - 空间复杂度：O(1)

## 21. 搜索二维矩阵 II
```
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
 
示例 1：
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true

示例 2：
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```
![搜索二维矩阵 II-示例1](/相关图册/搜索二维矩阵%20II-示例1.jpg)  
![搜索二维矩阵 II-示例2](/相关图册/搜索二维矩阵%20II-示例2.jpg) 
- 题解
    - 直接查找
        ```
        class Solution {
            public boolean searchMatrix(int[][] matrix, int target) {
                for (int[] row : matrix) {
                    for (int element : row) {
                        if (element == target) {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
        ```
        - 时间复杂度：O(mn)  
        - 空间复杂度：O(1)
    - 二分查找
        ```
        class Solution {
            public boolean searchMatrix(int[][] matrix, int target) {
                for (int[] row : matrix) {
                    int index = search(row, target);
                    if (index >= 0) {
                        return true;
                    }
                }
                return false;
            }

            public int search(int[] nums, int target) {
                int low = 0, high = nums.length - 1;
                while (low <= high) {
                    int mid = (high - low) / 2 + low;
                    int num = nums[mid];
                    if (num == target) {
                        return mid;
                    } else if (num > target) {
                        high = mid - 1;
                    } else {
                        low = mid + 1;
                    }
                }
                return -1;
            }
        }
        ```
        - 时间复杂度：O(mlogn)  
        - 空间复杂度：O(1)

## 22. 相交链表
```
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

图示两个链表在节点 c1 开始相交：

题目数据 保证 整个链式结构中不存在环。

注意，函数返回结果后，链表必须 保持其原始结构 。

示例 1：
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
— 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。
 
示例 2：
输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

示例 3：
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：No intersection
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。
```
![相交链表-示例1](/相关图册/相交链表-示例1.png)  
![相交链表-示例2](/相关图册/相交链表-示例2.png)   
![相交链表-示例3](/相关图册/相交链表-示例3.png) 
- 题解
    - 哈希集合
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        */
        public class Solution {
            public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
                Set<ListNode> visited = new HashSet<ListNode>();
                ListNode temp = headA;
                while (temp != null) {
                    visited.add(temp);
                    temp = temp.next;
                }
                temp = headB;
                while (temp != null) {
                    if (visited.contains(temp)) {
                        return temp;
                    }
                    temp = temp.next;
                }
                return null;
            }
        }
        ```
        - 时间复杂度：O(m+n)  
        - 空间复杂度：O(m)
    - 双指针
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        */
        public class Solution {
            public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
                if (headA == null || headB == null) {
                    return null;
                }
                ListNode pA = headA, pB = headB;
                while (pA != pB) {
                    pA = pA == null ? headB : pA.next;
                    pB = pB == null ? headA : pB.next;
                }
                return pA;
            }
        }
        ```
        - 时间复杂度：O(m+n)  
        - 空间复杂度：O(1)

## 23. 反转链表
```
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
 

示例 1：
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

示例 2：
输入：head = [1,2]
输出：[2,1]

示例 3：
输入：head = []
输出：[]
```
![反转链表-示例1](/相关图册/反转链表-示例1.jpg)  
![反转链表-示例2](/相关图册/反转链表-示例2.jpg)
- 题解
    - 迭代
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode reverseList(ListNode head) {
                ListNode prev = null;
                ListNode curr = head;
                while (curr != null) {
                    ListNode next = curr.next;
                    curr.next = prev;
                    prev = curr;
                    curr = next;
                }
                return prev;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)
    - 递归
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode reverseList(ListNode head) {
                if (head == null || head.next == null) {
                    return head;
                }
                ListNode newHead = reverseList(head.next);
                head.next.next = head;
                head.next = null;
                return newHead;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 24. 回文链表
```
给你一个单链表的头节点 head ，请你判断该链表是否为
回文链表
。如果是，返回 true ；否则，返回 false 。

示例 1：
输入：head = [1,2,2,1]
输出：true

示例 2：
输入：head = [1,2]
输出：false
```
![回文链表-示例1](/相关图册/回文链表-示例1.jpg)  
![回文链表-示例2](/相关图册/回文链表-示例2.jpg)
- 题解
    - 将值复制到数组中后用双指针法
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public boolean isPalindrome(ListNode head) {
                List<Integer> vals = new ArrayList<Integer>();

                // 将链表的值复制到数组中
                ListNode currentNode = head;
                while (currentNode != null) {
                    vals.add(currentNode.val);
                    currentNode = currentNode.next;
                }

                // 使用双指针判断是否回文
                int front = 0;
                int back = vals.size() - 1;
                while (front < back) {
                    if (!vals.get(front).equals(vals.get(back))) {
                        return false;
                    }
                    front++;
                    back--;
                }
                return true;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 快慢指针
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public boolean isPalindrome(ListNode head) {
                if (head == null) {
                    return true;
                }

                // 找到前半部分链表的为节点并反转后半部分链表
                ListNode firstHalfEnd = endOfFirstHalf(head);
                ListNode secondHalfStart = reverseList(firstHalfEnd.next);

                // 判断是否回文
                ListNode p1 = head;
                ListNode p2 = secondHalfStart;
                boolean result = true;
                while (result && p2 != null) {
                    if (p1.val != p2.val) {
                        result = false;
                    }
                    p1 = p1.next;
                    p2 = p2.next;
                }

                // 还原链表并返回结果
                firstHalfEnd.next = reverseList(secondHalfStart);
                return result;
            }

            private ListNode reverseList(ListNode head) {
                ListNode prev = null;
                ListNode curr = head;
                while (curr != null) {
                    ListNode nextTemp = curr.next;
                    curr.next = prev;
                    prev = curr;
                    curr = nextTemp;
                }
                return prev;
            }

            private ListNode endOfFirstHalf(ListNode head) {
                ListNode fast = head;
                ListNode slow = head;
                while (fast.next != null && fast.next.next != null) {
                    fast = fast.next.next;
                    slow = slow.next;
                }
                return slow;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 25. 环形链表
```
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```
![环形链表-示例1](/相关图册/环形链表-示例1.png)  
![环形链表-示例2](/相关图册/环形链表-示例2.png)  
![环形链表-示例3](/相关图册/环形链表-示例3.png)
- 题解
    - 哈希表
        ```
        /**
        * Definition for singly-linked list.
        * class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        */
        public class Solution {
            public boolean hasCycle(ListNode head) {
                Set<ListNode> seen = new HashSet<ListNode>();
                while (head != null) {
                    if (!seen.add(head)) {
                        return true;
                    }
                    head = head.next;
                }
                return false;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)
    - 快慢指针
        ```
        /**
        * Definition for singly-linked list.
        * class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        */
        public class Solution {
            public boolean hasCycle(ListNode head) {
                if (head == null || head.next == null) {
                    return false;
                }
                ListNode slow = head;
                ListNode fast = head.next;
                while (slow != fast) {
                    if (fast == null || fast.next == null) {
                        return false;
                    }
                    slow = slow.next;
                    fast = fast.next.next;
                }
                return true;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)

## 26. 环形链表 II
```
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

不允许修改 链表。

示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```
![环形链表 II-示例1](/相关图册/环形链表%20II-示例1.png)  
![环形链表 II-示例2](/相关图册/环形链表%20II-示例2.png)  
![环形链表 II-示例3](/相关图册/环形链表%20II-示例3.png)
- 题解
    - 哈希表
        ```
        /**
        * Definition for singly-linked list.
        * class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        */
        public class Solution {
            public ListNode detectCycle(ListNode head) {
            ListNode pos = head;
            Set<ListNode> visited = new HashSet<ListNode>();
            while (pos != null) {
                if (visited.contains(pos)) {
                    return pos;
                } else {
                    visited.add(pos);
                }
                pos = pos.next;
            }  
            return null;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)
    - 快慢指针
        ```
        /**
        * Definition for singly-linked list.
        * class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        */
        public class Solution {
            public ListNode detectCycle(ListNode head) {
            if (head == null) {
                return null;
            }
            ListNode slow = head, fast = head;
            while (fast != null) {
                slow = slow.next;
                if (fast.next != null) {
                    fast = fast.next.next;
                } else {
                    return null;
                } 
                if (fast == slow) {
                    ListNode ptr = head;
                    while (ptr != slow) {
                        ptr = ptr.next;
                        slow = slow.next;
                    }
                    return ptr;
                }
            }
            return null;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(1)

## 27. 合并两个有序链表
```
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例 1：
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

示例 2：
输入：l1 = [], l2 = []
输出：[]

示例 3：
输入：l1 = [], l2 = [0]
输出：[0]
```
![合并两个有序链表-示例1](/相关图册/合并两个有序链表-示例1.jpg)  
- 题解
    - 递归
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
                if (l1 == null) {
                    return l2;
                } else if (l2 == null) {
                    return l1;
                } else if (l1.val < l2.val) {
                    l1.next = mergeTwoLists(l1.next, l2);
                    return l1;
                } else {
                    l2.next = mergeTwoLists(l1, l2.next);
                    return l2;
                }
            }
        }
        ```
        - 时间复杂度：O(n+m)  
        - 空间复杂度：O(n+m)

    - 迭代
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
                ListNode prehead = new ListNode(-1);

                ListNode prev = prehead;
                while (l1 != null && l2 != null) {
                    if (l1.val <= l2.val) {
                        prev.next = l1;
                        l1 = l1.next;
                    } else {
                        prev.next = l2;
                        l2 = l2.next;
                    }
                    prev = prev.next;
                }

                // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
                prev.next = l1 == null ? l2 : l1;

                return prehead.next;
            }
        }
        ```
        - 时间复杂度：O(n+m)  
        - 空间复杂度：O(1)

## 28. 两数相加
```
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例 1：
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

示例 2：
输入：l1 = [0], l2 = [0]
输出：[0]

示例 3：
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```
![两数相加-示例1](/相关图册/两数相加-示例1.jpg)
- 题解
    - 模拟
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
                ListNode head = null, tail = null;
                int carry = 0;
                while (l1 != null || l2 != null) {
                    int n1 = l1 != null ? l1.val : 0;
                    int n2 = l2 != null ? l2.val : 0;
                    int sum = n1 + n2 + carry;
                    if  (head == null) {
                        head = tail = new ListNode(sum % 10);
                    } else {
                        tail.next = new ListNode(sum % 10);
                        tail = tail.next;
                    }
                    carry = sum / 10;
                    if (l1 != null) {
                        l1 = l1.next;
                    }
                    if (l2 !=null) {
                        l2 = l2.next;
                    }
                }
                if (carry > 0) {
                    tail.next = new ListNode(carry);
                }
                return head;
            }
        }
        ```
        - 时间复杂度：O(max(m,n))  
        - 空间复杂度：O(1)

## 29. 删除链表的倒数第 N 个结点
```
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

示例 1：
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

示例 2：
输入：head = [1], n = 1
输出：[]

示例 3：
输入：head = [1,2], n = 1
输出：[1]
```
![删除链表的倒数第 N 个结点-示例1](/相关图册/删除链表的倒数第%20N%20个结点-示例1.jpg)
- 题解
    - 哑巴节点
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode removeNthFromEnd(ListNode head, int n) {
                ListNode dummy = new ListNode(0, head);
                int length = getLength(head);
                ListNode cur = dummy;
                for (int i = 1; i < length - n + 1; ++i) {
                    cur = cur.next;
                }
                cur.next = cur.next.next;
                ListNode ans = dummy.next;
                return ans;
            }

            public int getLength(ListNode head) {
                int length = 0;
                while (head != null) {
                    ++length;
                    head = head.next;
                }
                return length;
            }
        }
        ```
        - 时间复杂度：O(L)  
        - 空间复杂度：O(1)
    - 快慢指针
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode removeNthFromEnd(ListNode head, int n) {
                ListNode dummy = new ListNode(0, head);
                ListNode first = head;
                ListNode second = dummy;
                for (int i = 0; i < n; ++i) {
                    first = first.next;
                }
                while (first != null) {
                    first = first.next;
                    second = second.next;
                }
                second.next = second.next.next;
                ListNode ans = dummy.next;
                return ans;
            }
        }
        ```
        - 时间复杂度：O(L)  
        - 空间复杂度：O(1)

## 30. 两两交换链表中的节点
```
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

示例 1：
输入：head = [1,2,3,4]
输出：[2,1,4,3]

示例 2：
输入：head = []
输出：[]

示例 3：
输入：head = [1]
输出：[1]
```
![两两交换链表中的节点-示例1](/相关图册/两两交换链表中的节点-示例1.jpg)
- 题解
    - 递归
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode swapPairs(ListNode head) {
                if (head == null || head.next == null) {
                    return head;
                }
                ListNode newHead = head.next;
                head.next = swapPairs(newHead.next);
                newHead.next = head;
                return newHead;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 迭代
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode swapPairs(ListNode head) {
                ListNode dummyHead = new ListNode(0);
                dummyHead.next = head;
                ListNode temp = dummyHead;
                while (temp.next != null && temp.next.next != null) {
                    ListNode node1 = temp.next;
                    ListNode node2 = temp.next.next;
                    temp.next = node2;
                    node1.next = node2.next;
                    node2.next = node1;
                    temp = node1;
                }
                return dummyHead.next;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 31. K 个一组翻转链表
```
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

示例 1：
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]

示例 2：
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```
![K 个一组翻转链表-示例1](/相关图册/K%20个一组翻转链表-示例1.jpg)  
![K 个一组翻转链表-示例2](/相关图册/K%20个一组翻转链表-示例2.jpg)
- 题解
    - 模拟
        ```
        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     int val;
        *     ListNode next;
        *     ListNode() {}
        *     ListNode(int val) { this.val = val; }
        *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
        * }
        */
        class Solution {
            public ListNode reverseKGroup(ListNode head, int k) {
                ListNode hair = new ListNode(0);
                hair.next = head;
                ListNode pre = hair;

                while (head != null) {
                    ListNode tail = pre;
                    // 查看剩余部分成都是否大于等于 k
                    for (int i = 0; i < k; ++i) {
                        tail = tail.next;
                        if (tail == null) {
                            return hair.next;
                        }
                    }
                    ListNode nex = tail.next;
                    ListNode[] reverse = myReverse(head, tail);
                    head = reverse[0];
                    tail = reverse[1];
                    // 把子链表重新接回原链表
                    pre.next = head;
                    tail.next = nex;
                    pre = tail;
                    head = tail.next;
                }

                return hair.next;
            }

            public ListNode[] myReverse(ListNode head, ListNode tail) {
                ListNode prev = tail.next;
                ListNode p = head;
                while (prev != tail) {
                    ListNode nex = p.next;
                    p.next = prev;
                    prev = p;
                    p = nex;
                }
                return new ListNode[]{tail, head};
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 32. 随机链表的复制
```
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。

你的代码 只 接受原链表的头节点 head 作为传入参数。

示例 1：
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

示例 2：
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]

示例 3：
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```
![随机链表的复制-示例1](/相关图册/随机链表的复制-示例1.png)  
![随机链表的复制-示例2](/相关图册/随机链表的复制-示例2.png)
![随机链表的复制-示例3](/相关图册/随机链表的复制-示例3.png)  
- 题解
    - 回溯 + 哈希表
        ```
        /*
        // Definition for a Node.
        class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }
        */

        class Solution {
            Map<Node, Node> cachedNode = new HashMap<Node, Node>();

            public Node copyRandomList(Node head) {
                if (head == null) {
                    return null;
                }
                if (!cachedNode.containsKey(head)) {
                    Node headNew = new Node(head.val);
                    cachedNode.put(head, headNew);
                    headNew.next = copyRandomList(head.next);
                    headNew.random = copyRandomList(head.random);
                }
                return cachedNode.get(head);
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 迭代 + 节点拆分
        ```
        /*
        // Definition for a Node.
        class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }
        */

        class Solution {
            public Node copyRandomList(Node head) {
                if (head == null) {
                    return null;
                }
                for (Node node = head; node != null; node = node.next.next) {
                    Node nodeNew = new Node(node.val);
                    nodeNew.next = node.next;
                    node.next = nodeNew; 
                }
                for (Node node = head; node != null; node = node.next.next) {
                    Node nodeNew = node.next;
                    nodeNew.random = (node.random != null) ? node.random.next : null;
                }
                Node headNew = head.next;
                for (Node node = head; node != null; node = node.next) {
                    Node nodeNew = node.next;
                    node.next = node.next.next;
                    nodeNew.next = (nodeNew.next != null) ? nodeNew.next.next : null;
                }
                return headNew;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)


