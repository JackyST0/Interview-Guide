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
    
## 20. 旋转图像
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

## 33. 排序链表
```
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

示例 1：
输入：head = [4,2,1,3]
输出：[1,2,3,4]

示例 2：
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]

示例 3：
输入：head = []
输出：[]
```
![排序链表-示例1](/相关图册/排序链表-示例1.jpg)       
![排序链表-示例2](/相关图册/排序链表-示例2.jpg)  
- 题解
    - 自顶向下归并排序
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
            public ListNode sortList(ListNode head) {
                return sortList(head, null);
            }

            public ListNode sortList(ListNode head, ListNode tail) {
                if (head == null) {
                    return head;
                }
                if (head.next == tail) {
                    head.next = null;
                    return head;
                }
                ListNode slow = head, fast = head;
                while (fast != tail) {
                    slow = slow.next;
                    fast = fast.next;
                    if (fast != tail) {
                        fast = fast.next;
                    }
                }
                ListNode mid = slow;
                ListNode list1 = sortList(head, mid);
                ListNode list2 = sortList(mid, tail);
                ListNode sorted = merge(list1, list2);
                return sorted;
            }

            public ListNode merge(ListNode head1, ListNode head2) {
                ListNode dummyHead = new ListNode(0);
                ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
                while (temp1 != null && temp2 != null) {
                    if (temp1.val <= temp2.val) {
                        temp.next = temp1;
                        temp1 = temp1.next;
                    } else {
                        temp.next = temp2;
                        temp2 = temp2.next;
                    }
                    temp = temp.next;
                }
                if (temp1 != null) {
                    temp.next = temp1;
                } else if (temp2 != null) {
                    temp.next = temp2;
                }
                return dummyHead.next;
            }
        }
        ```
        - 时间复杂度：O(nlogn)  
        - 空间复杂度：O(logn)
    - 自底向上归并排序
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
            public ListNode sortList(ListNode head) {
                if (head == null) {
                    return head;
                }
                int length = 0;
                ListNode node = head;
                while (node != null) {
                    length++;
                    node = node.next;
                }
                ListNode dummyHead = new ListNode(0, head);
                for (int subLength = 1; subLength < length; subLength <<= 1) {
                    ListNode prev = dummyHead, curr = dummyHead.next;
                    while (curr != null) {
                        ListNode head1 = curr;
                        for (int i = 1; i < subLength && curr.next != null; i++) {
                            curr = curr.next;
                        }
                        ListNode head2 = curr.next;
                        curr.next = null; 
                        curr = head2;
                        for (int i = 1; i < subLength && curr != null && curr.next != null; i++) {
                            curr = curr.next;
                        }
                        ListNode next = null;
                        if (curr != null) {
                            next = curr.next;
                            curr.next = null;
                        }
                        ListNode merged = merge(head1, head2);
                        prev.next = merged;
                        while (prev.next != null) {
                            prev = prev.next;
                        }
                        curr = next;
                    }
                }
                return dummyHead.next;
            }

            public ListNode merge(ListNode head1, ListNode head2) {
                ListNode dummyHead = new ListNode(0);
                ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
                while (temp1 != null && temp2 != null) {
                    if (temp1.val <= temp2.val) {
                        temp.next = temp1;
                        temp1 = temp1.next;
                    } else {
                        temp.next = temp2;
                        temp2 = temp2.next;
                    }
                    temp = temp.next;
                }
                if (temp1 != null) {
                    temp.next = temp1;
                } else if (temp2 != null) {
                    temp.next = temp2;
                }
                return dummyHead.next;
            }
        }
        ```
        - 时间复杂度：O(nlogn)  
        - 空间复杂度：O(1)

## 34. 合并 K 个升序链表
```
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

示例 1：
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6

示例 2：
输入：lists = []
输出：[]

示例 3：
输入：lists = [[]]
输出：[]
```
- 题解
    - 顺序合并
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
            public ListNode mergeKLists(ListNode[] lists) {
                ListNode ans = null;
                for (int i = 0; i < lists.length; ++i) {
                    ans = mergeTwoLists(ans, lists[i]);
                }
                return ans;
            }

            public ListNode mergeTwoLists(ListNode a, ListNode b) {
                if (a == null || b == null) {
                    return a != null ? a : b;
                }
                ListNode head = new ListNode(0);
                ListNode tail = head, aPtr = a, bPtr = b;
                while (aPtr != null && bPtr != null) {
                    if (aPtr.val < bPtr.val) {
                        tail.next = aPtr;
                        aPtr = aPtr.next;
                    } else {
                        tail.next = bPtr;
                        bPtr = bPtr.next;
                    }
                    tail = tail.next;
                }
                tail.next = (aPtr != null ? aPtr : bPtr);
                return head.next;
            }
        }
        ```
        - 时间复杂度：O(k²n)  
        - 空间复杂度：O(1)
    - 分治合并
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
            public ListNode mergeKLists(ListNode[] lists) {
                return merge(lists, 0, lists.length - 1);
            }

            public ListNode merge(ListNode[] lists, int l, int r) {
                if (l == r) {
                    return lists[l];
                }
                if (l > r) {
                    return null;
                }
                int mid = (l + r) >> 1;
                return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 1, r));
            }

            public ListNode mergeTwoLists(ListNode a, ListNode b) {
                if (a == null || b == null) {
                    return a != null ? a : b;
                }
                ListNode head = new ListNode(0);
                ListNode tail = head, aPtr = a, bPtr = b;
                while (aPtr != null && bPtr != null) {
                    if (aPtr.val < bPtr.val) {
                        tail.next = aPtr;
                        aPtr = aPtr.next;
                    } else {
                        tail.next = bPtr;
                        bPtr = bPtr.next;
                    }
                    tail = tail.next;
                }
                tail.next = (aPtr != null ? aPtr : bPtr);
                return head.next;
            }
        }
        ```
        - 时间复杂度：O(kn×logk)  
        - 空间复杂度：O(logk)

## 35. LRU 缓存
```
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```
- 题解
    - 哈希表 + 双向链表
        ```
        class LRUCache {
            class DLinkedNode {
                int key;
                int value;
                DLinkedNode prev;
                DLinkedNode next;
                public DLinkedNode() {}
                public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
            }

            private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
            private int size;
            private int capacity;
            private DLinkedNode head, tail;

            public LRUCache(int capacity) {
                this.size = 0;
                this.capacity = capacity;
                // 使用伪头部和伪尾部节点
                head = new DLinkedNode();
                tail = new DLinkedNode();
                head.next = tail;
                tail.prev = head;
            }
            
            public int get(int key) {
                DLinkedNode node = cache.get(key);
                if (node == null) {
                    return -1;
                }
                // 如果 key 存在，先通过哈希表定位， 再移到头部
                moveToHead(node);
                return node.value;
            }
            
            public void put(int key, int value) {
                DLinkedNode node = cache.get(key);
                if (node == null) {
                    // 如果 key 不存在，创建一个新的节点
                    DLinkedNode newNode = new DLinkedNode(key, value);
                    // 添加进哈希表
                    cache.put(key, newNode);
                    // 添加至双向链表的头部
                    addToHead(newNode);
                    ++size;
                    if (size > capacity) {
                        // 如果超出容量，删除双向链表的尾部节点
                        DLinkedNode tail = removeTail();
                        // 删除哈希表中对应的项
                        cache.remove(tail.key);
                        --size;
                    }
                } else {
                    //如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
                    node.value = value;
                    moveToHead(node);
                }
            }

            private void addToHead(DLinkedNode node) {
                node.prev = head;
                node.next = head.next;
                head.next.prev = node;
                head.next = node;
            }

            private void removeNode(DLinkedNode node) {
                node.prev.next = node.next;
                node.next.prev = node.prev;
            }

            private void moveToHead(DLinkedNode node) {
                removeNode(node);
                addToHead(node);
            }

            private DLinkedNode removeTail() {
                DLinkedNode res = tail.prev;
                removeNode(res);
                return res;
            }
        }

        /**
        * Your LRUCache object will be instantiated and called as such:
        * LRUCache obj = new LRUCache(capacity);
        * int param_1 = obj.get(key);
        * obj.put(key,value);
        */
        ```
        - 时间复杂度：O(1)  
        - 空间复杂度：O(capacity)

## 36. 二叉树的中序遍历
```
给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

示例 1：
输入：root = [1,null,2,3]
输出：[1,3,2]

示例 2：
输入：root = []
输出：[]

示例 3：
输入：root = [1]
输出：[1]
```
![二叉树的中序遍历-示例1](/相关图册/二叉树的中序遍历-示例1.jpg)      
- 题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public List<Integer> inorderTraversal(TreeNode root) {
            List<Integer> res = new ArrayList<Integer>();
            inorder(root, res);
            return res; 
            }

            public void inorder(TreeNode root,  List<Integer> res) {
                if (root == null) {
                    return;
                }
                inorder(root.left, res);
                res.add(root.val);
                inorder(root.right, res);
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 迭代
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public List<Integer> inorderTraversal(TreeNode root) {
                List<Integer> res = new ArrayList<Integer>();
                Deque<TreeNode> stk = new LinkedList<TreeNode>();
                while (root != null || !stk.isEmpty()) {
                    while (root != null) {
                        stk.push(root);
                        root = root.left;
                    }
                    root = stk.pop();
                    res.add(root.val);
                    root = root.right;
                }
                return res; 
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - Morris 中序遍历
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public List<Integer> inorderTraversal(TreeNode root) {
                List<Integer> res = new ArrayList<Integer>();
                TreeNode predecessor = null;

                while (root != null) {
                    if (root.left != null) {
                        // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
                        predecessor = root.left;
                        while (predecessor.right != null && predecessor.right != root) {
                            predecessor = predecessor.right;
                        }

                        // 让 predecessor 的右指针指向 root，继续遍历左子树
                        if (predecessor.right == null) {
                            predecessor.right = root;
                            root = root.left;
                        }
                        // 说明左子树已经访问完了，我们需要断开链接
                        else {
                            res.add(root.val);
                            predecessor.right = null;
                            root = root.right;
                        }
                    }
                    // 如果没有左孩子，则直接访问右孩子
                    else {
                        res.add(root.val);
                        root = root.right;
                    }
                }
                return res; 
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 37. 二叉树的最大深度
```
给定一个二叉树 root ，返回其最大深度。

二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。

示例 1：
输入：root = [3,9,20,null,null,15,7]
输出：3

示例 2：
输入：root = [1,null,2]
输出：2
```
![二叉树的最大深度-示例1](/相关图册/二叉树的最大深度-示例1.jpg)  
- 题解
    - 深度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public int maxDepth(TreeNode root) {
                if (root == null) {
                    return 0;
                } else {
                    int leftHeight = maxDepth(root.left);
                    int rightHeight = maxDepth(root.right);
                    return Math.max(leftHeight, rightHeight) + 1;
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(height)
    - 广度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public int maxDepth(TreeNode root) {
                if (root == null) {
                    return 0;
                } 
                Queue<TreeNode> queue = new LinkedList<TreeNode>();
                queue.offer(root);
                int ans = 0;
                while (!queue.isEmpty()) {
                    int size = queue.size();
                    while (size > 0) {
                        TreeNode node = queue.poll();
                        if (node.left != null) {
                            queue.offer(node.left);
                        }
                        if (node.right != null) {
                            queue.offer(node.right);
                        }
                        size--;
                    }
                    ans++;
                } 
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 38. 翻转二叉树
```
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

示例 1：
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]

示例 2：
输入：root = [2,1,3]
输出：[2,3,1]

示例 3：
输入：root = []
输出：[]
```
![翻转二叉树-示例1](/相关图册/翻转二叉树-示例1.jpg)  
![翻转二叉树-示例2](/相关图册/翻转二叉树-示例2.jpg)  
- 题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public TreeNode invertTree(TreeNode root) {
                if (root == null) {
                    return null;
                }
                TreeNode left = invertTree(root.left);
                TreeNode right = invertTree(root.right);
                root.left = right;
                root.right = left;
                return root;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)

## 39. 对称二叉树
```
给你一个二叉树的根节点 root ， 检查它是否轴对称。

示例 1：
输入：root = [1,2,2,3,4,4,3]
输出：true

示例 2：
输入：root = [1,2,2,null,3,null,3]
输出：false
```
![对称二叉树-示例1](/相关图册/对称二叉树-示例1.webp)  
![对称二叉树-示例2](/相关图册/对称二叉树-示例2.webp)
- 题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public boolean isSymmetric(TreeNode root) {
                return check(root.left, root.right);
            }

            public boolean check(TreeNode p, TreeNode q) {
                if (p == null && q == null) {
                    return true;
                }
                if (p == null || q == null) {
                    return false;
                }
                return p.val == q.val && check(p.left, q.right) && check(p.right, q.left); 
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 迭代
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public boolean isSymmetric(TreeNode root) {
                return check(root, root);
            }

            public boolean check(TreeNode u, TreeNode v) {
                Queue<TreeNode> q = new LinkedList<TreeNode>();
                q.offer(u);
                q.offer(v);
                while (!q.isEmpty()) {
                    u = q.poll();
                    v = q.poll();
                    if (u == null && v == null) {
                        continue;
                    }
                    if ((u == null || v == null) || (u.val != v.val)) {
                        return false;
                    }

                    q.offer(u.left);
                    q.offer(v.right);

                    q.offer(u.right);
                    q.offer(v.left);
                }
                return true;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 40. 二叉树的直径
```
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

两节点之间路径的 长度 由它们之间边数表示。

示例 1：
输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。

示例 2：
输入：root = [1,2]
输出：1
```
![二叉树的直径-示例1](/相关图册/二叉树的直径-示例1.jpg)  
- 题解
    - 深度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            int ans;

            public int diameterOfBinaryTree(TreeNode root) {
                ans = 1;
                depth(root);
                return ans - 1;   
            }
            public int depth(TreeNode node) {
                if (node == null) {
                    return 0; // 访问到空节点了，返回0
                }
                int L = depth(node.left); // 左儿子为根的子树的深度
                int R = depth(node.right); // 右儿子为根的子树的深度
                ans = Math.max(ans, L + R + 1); // 计算 d_node 即 L+R+1 并更新 ans
                return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(Height)

## 41. 二叉树的层序遍历
```
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

示例 1：
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]

示例 2：
输入：root = [1]
输出：[[1]]

示例 3：
输入：root = []
输出：[]
```
![二叉树的层序遍历-示例1](/相关图册/二叉树的层序遍历-示例1.jpg)  
- 题解
    - 广度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public List<List<Integer>> levelOrder(TreeNode root) {
                List<List<Integer>> ret = new ArrayList<List<Integer>>();
                if (root == null) {
                    return ret;
                }

                Queue<TreeNode> queue = new LinkedList<TreeNode>();
                queue.offer(root);
                while (!queue.isEmpty()) {
                    List<Integer> level = new ArrayList<Integer>();
                    int currentLevelSize = queue.size();
                    for (int i = 1; i <= currentLevelSize; ++i) {
                        TreeNode node = queue.poll();
                        level.add(node.val);
                        if (node.left != null) {
                            queue.offer(node.left);
                        }
                        if (node.right != null) {
                            queue.offer(node.right);
                        }
                    }
                    ret.add(level);
                } 

                return ret;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 42. 将有序数组转换为二叉搜索树
```
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 平衡 二叉搜索树。

示例 1：
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：

示例 2：
输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```
![将有序数组转换为二叉搜索树-示例1(输入)](/相关图册/将有序数组转换为二叉搜索树-示例1(输入).jpg)     
![将有序数组转换为二叉搜索树-示例1(输出)](/相关图册/将有序数组转换为二叉搜索树-示例1(输出).jpg)     
![将有序数组转换为二叉搜索树-示例2](/相关图册/将有序数组转换为二叉搜索树-示例2.jpg)
- 题解
    - 中序遍历，总是选择中间位置左边的数字作为根节点
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public TreeNode sortedArrayToBST(int[] nums) {
                return helper(nums, 0, nums.length - 1);
            }

            public TreeNode helper(int[] nums, int left, int right) {
                if (left > right) {
                    return null;
                }

                // 总是选择中间位置左边的数字作为根节点
                int mid = (left + right) / 2;

                TreeNode root = new TreeNode(nums[mid]);
                root.left = helper(nums, left, mid - 1);
                root.right = helper(nums, mid + 1,right);
                return root;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(logn)
    - 中序遍历，总是选择中间位置右边的数字作为根节点
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public TreeNode sortedArrayToBST(int[] nums) {
                return helper(nums, 0, nums.length - 1);
            }

            public TreeNode helper(int[] nums, int left, int right) {
                if (left > right) {
                    return null;
                }

                // 总是选择中间位置左边的数字作为根节点
                int mid = (left + right + 1) / 2;

                TreeNode root = new TreeNode(nums[mid]);
                root.left = helper(nums, left, mid - 1);
                root.right = helper(nums, mid + 1,right);
                return root;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(logn)
  
## 43. 验证二叉搜索树
```
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

有效 二叉搜索树定义如下：

节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
 
示例 1：
输入：root = [2,1,3]
输出：true

示例 2：
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
```
![验证二叉搜索树-示例1](/相关图册/验证二叉搜索树-示例1.jpg)     
![验证二叉搜索树-示例2](/相关图册/验证二叉搜索树-示例2.jpg)
- 题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public boolean isValidBST(TreeNode root) {
                return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
            }

            public boolean isValidBST(TreeNode node, long lower, long upper) {
                if (node == null) {
                    return true;
                }
                if (node.val <= lower || node.val >= upper) {
                    return false;
                }
                return isValidBST(node.left, lower, node.val) && isValidBST(node.right, node.val, upper);
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 中序遍历
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public boolean isValidBST(TreeNode root) {
                return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
            }

            public boolean isValidBST(TreeNode node, long lower, long upper) {
                if (node == null) {
                    return true;
                }
                if (node.val <= lower || node.val >= upper) {
                    return false;
                }
                return isValidBST(node.left, lower, node.val) && isValidBST(node.right, node.val, upper);
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 44. 二叉搜索树中第 K 小的元素
```
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。

示例 1：
输入：root = [3,1,4,null,2], k = 1
输出：1

示例 2：
输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
```
![二叉搜索树中第 K 小的元素-示例1](/相关图册/二叉搜索树中第%20K%20小的元素-示例1.jpg)   
![二叉搜索树中第 K 小的元素-示例2](/相关图册/二叉搜索树中第%20K%20小的元素-示例2.jpg)    
- 题解
    - 中序遍历
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public int kthSmallest(TreeNode root, int k) {
                Deque<TreeNode> stack = new ArrayDeque<TreeNode>();
                while (root != null || !stack.isEmpty()) {
                    while (root != null) {
                        stack.push(root);
                        root = root.left;
                    }
                    root = stack.pop();
                    --k;
                    if (k == 0) {
                        break;
                    }
                    root = root.right;
                }
                return root.val;
            }
        }
        ```
        - 时间复杂度：O(H+k)  
        - 空间复杂度：O(H)

## 45. 二叉树的右视图
```
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

示例 1：
输入：root = [1,2,3,null,5,null,4]
输出：[1,3,4]
解释：

示例 2：
输入：root = [1,2,3,4,null,null,null,5]
输出：[1,3,4,5]
解释：

示例 3：
输入：root = [1,null,3]
输出：[1,3]

示例 4：
输入：root = []
输出：[]
```
![二叉树的右视图-示例1](/相关图册/二叉树的右视图-示例1.png)   
![二叉树的右视图-示例2](/相关图册/二叉树的右视图-示例2.png)  
- 题解
    - 深度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public List<Integer> rightSideView(TreeNode root) {
                Map<Integer, Integer> rightmostValueAtDepth = new HashMap<Integer, Integer>();
                int max_depth = -1;

                Deque<TreeNode> nodeStack = new LinkedList<TreeNode>();
                Deque<Integer> depthStack = new LinkedList<Integer>();
                nodeStack.push(root);
                depthStack.push(0);

                while (!nodeStack.isEmpty()) {
                    TreeNode node = nodeStack.pop();
                    int depth = depthStack.pop();

                    if (node != null) {
                        // 维护二叉树的最大深度
                        max_depth = Math.max(max_depth, depth);

                        // 如果不存在对应深度的节点我们才插入
                        if (!rightmostValueAtDepth.containsKey(depth)) {
                            rightmostValueAtDepth.put(depth, node.val);
                        }

                        nodeStack.push(node.left);
                        nodeStack.push(node.right);
                        depthStack.push(depth + 1);
                        depthStack.push(depth + 1);
                    } 
                }

                List<Integer> rightView = new ArrayList<Integer>();
                for (int depth = 0; depth <= max_depth; depth++) {
                    rightView.add(rightmostValueAtDepth.get(depth));
                }

                return rightView;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 广度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public List<Integer> rightSideView(TreeNode root) {
                Map<Integer, Integer> rightmostValueAtDepth = new HashMap<Integer, Integer>();
                int max_depth = -1;

                Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
                Queue<Integer> depthQueue = new LinkedList<Integer>();
                nodeQueue.add(root);
                depthQueue.add(0);

                while (!nodeQueue.isEmpty()) {
                    TreeNode node = nodeQueue.remove();
                    int depth = depthQueue.remove();

                    if (node != null) {
                        // 维护二叉树的最大深度
                        max_depth = Math.max(max_depth, depth);

                        // 由于每一层最后一个访问到的节点才是我们要的答案，因此不断更新对应深度的信息即可
                        rightmostValueAtDepth.put(depth, node.val);

                        nodeQueue.add(node.left);
                        nodeQueue.add(node.right);
                        depthQueue.add(depth + 1);
                        depthQueue.add(depth + 1);
                    } 
                }

                List<Integer> rightView = new ArrayList<Integer>();
                for (int depth = 0; depth <= max_depth; depth++) {
                    rightView.add(rightmostValueAtDepth.get(depth));
                }

                return rightView;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 46. 二叉树展开为链表
```
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

示例 1：
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]

示例 2：
输入：root = []
输出：[]

示例 3：
输入：root = [0]
输出：[0]
```
![二叉树展开为链表-示例1](/相关图册/二叉树展开为链表-示例1.jpg)  
- 题解
    - 前序遍历(递归) 
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public void flatten(TreeNode root) {
                List<TreeNode> list = new ArrayList<TreeNode>();
                preorderTraversal(root, list);
                int size = list.size();
                for (int i = 1; i < size; i++) {
                    TreeNode prev = list.get(i - 1), curr = list.get(i);
                    prev.left = null;
                    prev.right = curr;
                }
            }

            public void preorderTraversal(TreeNode root, List<TreeNode> list) {
                if (root != null) {
                    list.add(root);
                    preorderTraversal(root.left, list);
                    preorderTraversal(root.right, list);
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 前序遍历(迭代)
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public void flatten(TreeNode root) {
                List<TreeNode> list = new ArrayList<TreeNode>();
                Deque<TreeNode> stack = new LinkedList<TreeNode>();
                TreeNode node = root;
                while (node != null || !stack.isEmpty()) {
                    while (node != null) {
                        list.add(node);
                        stack.push(node);
                        node = node.left;
                    }
                    node = stack.pop();
                    node = node.right;
                }
                int size = list.size();
                for (int i = 1; i < size; i++) {
                    TreeNode prew = list.get(i - 1), curr = list.get(i);
                    prew.left = null;
                    prew.right = curr;
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 寻找前驱节点 
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public void flatten(TreeNode root) {
                TreeNode curr = root;
                while (curr != null) {
                    if (curr.left != null) {
                        TreeNode next = curr.left;
                        TreeNode predecessor = next;
                        while (predecessor.right != null) {
                            predecessor = predecessor.right;
                        }
                        predecessor.right = curr.right;
                        curr.left = null;
                        curr.right = next;
                    }
                    curr = curr.right;
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 47. 从前序与中序遍历序列构造二叉树
```
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

示例 1:
输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]

示例 2:
输入: preorder = [-1], inorder = [-1]
输出: [-1]
```
![从前序与中序遍历序列构造二叉树-示例1](/相关图册/从前序与中序遍历序列构造二叉树-示例1.jpg)  
- 题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            private Map<Integer, Integer> indexMap;

            public TreeNode myBuildtree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
                if (preorder_left > preorder_right) {
                    return null;
                }

                // 前序遍历中的第一个节点就是根节点
                int preorder_root = preorder_left;
                // 在中序遍历中定位根节点
                int inorder_root = indexMap.get(preorder[preorder_root]);

                // 先把根节点建立出来
                TreeNode root = new TreeNode(preorder[preorder_root]);
                // 得到左子树中的节点数目
                int size_left_subtree = inorder_root - inorder_left;
                // 递归地构造左子树，并连接到根节点
                // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
                root.left = myBuildtree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
                // 递归地构造右子树，并连接到根节点
                // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了需要中序遍历中「从 根节点定位+1 到 右边界」的元素
                root.right = myBuildtree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
                return root;
            }

            public TreeNode buildTree(int[] preorder, int[] inorder) {
                int n = preorder.length;
                // 构造哈希映射，帮助我们快速定位根节点
                indexMap = new HashMap<Integer, Integer>();
                for (int i = 0; i < n; i++) {
                    indexMap.put(inorder[i], i );
                }
                return myBuildtree(preorder, inorder, 0, n - 1, 0, n - 1);
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 48. 路径总和 III
```
给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

示例 1：
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。

示例 2：
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3
```
![路径总和 III-示例1](/相关图册/路径总和%20III-示例1.jpg)  
- 题解
    - 深度优先搜索
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            public int pathSum(TreeNode root, int targetSum) {
                if (root == null) {
                    return 0;
                }

                int ret = rootSum(root, targetSum);
                ret += pathSum(root.left, targetSum);
                ret += pathSum(root.right, targetSum);
                return ret;
            }

            public int rootSum(TreeNode root, long targetSum) {
                int ret = 0;
                if (root == null) {
                    return 0;
                }
                int val = root.val;
                if (val == targetSum) {
                    ret++;
                }

                ret += rootSum(root.left, targetSum - val);
                ret += rootSum(root.right, targetSum - val);
                return ret; 
            }
        }
        ```
        - 时间复杂度：O(N²)  
        - 空间复杂度：O(N)

## 49. 二叉树的最近公共祖先
```
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

示例 1：
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

示例 2：
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。

示例 3：
输入：root = [1,2], p = 1, q = 2
输出：1
```
![二叉树的最近公共祖先-示例1](/相关图册/二叉树的最近公共祖先-示例1.png)  
![二叉树的最近公共祖先-示例2](/相关图册/二叉树的最近公共祖先-示例2.png)  
- 题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode(int x) { val = x; }
        * }
        */
        class Solution {

            private TreeNode ans;

            public Solution() {
                this.ans = null;
            }

            private boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
                if (root == null) return false;
                boolean lson = dfs(root.left, p, q);
                boolean rson = dfs(root.right, p, q);
                if ((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))) {
                    ans = root;
                }
                return lson || rson || (root.val == p.val || root.val == q.val);
            }

            public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
                this.dfs(root, p , q);
                return this.ans;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)
    - 存储父节点
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode(int x) { val = x; }
        * }
        */
        class Solution {
            Map<Integer, TreeNode> parent = new HashMap<Integer, TreeNode>();
            Set<Integer> visited = new HashSet<Integer>();

            public void dfs(TreeNode root) {
                if (root.left != null) {
                    parent.put(root.left.val, root);
                    dfs(root.left);
                }
                if (root.right != null) {
                    parent.put(root.right.val, root);
                    dfs(root.right);
                }
            }

            public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
                dfs(root);
                while (p != null) {
                    visited.add(p.val);
                    p = parent.get(p.val);
                }
                while (q != null) {
                    if (visited.contains(q.val)) {
                        return q;
                    }
                    q = parent.get(q.val);
                }
                return null;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)

## 50. 二叉树中的最大路径和
```
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

示例 1：
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6

示例 2：
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```
![二叉树中的最大路径和-示例1](/相关图册/二叉树中的最大路径和-示例1.jpg)     
![二叉树中的最大路径和-示例2](/相关图册/二叉树中的最大路径和-示例2.jpg) 
-  题解
    - 递归
        ```
        /**
        * Definition for a binary tree node.
        * public class TreeNode {
        *     int val;
        *     TreeNode left;
        *     TreeNode right;
        *     TreeNode() {}
        *     TreeNode(int val) { this.val = val; }
        *     TreeNode(int val, TreeNode left, TreeNode right) {
        *         this.val = val;
        *         this.left = left;
        *         this.right = right;
        *     }
        * }
        */
        class Solution {
            int maxSum = Integer.MIN_VALUE;

            public int maxPathSum(TreeNode root) {
                maxGain(root);
                return maxSum;
            }

            public int maxGain(TreeNode node) {
                if (node == null) {
                    return 0;
                }

                // 递归计算左右子节点的最大贡献值
                // 只有在最大贡献值大于 0 时，才会选取对应子节点
                int leftGain = Math.max(maxGain(node.left), 0);
                int rightGain = Math.max(maxGain(node.right), 0);

                // 节点的最大路径和取决于该结点的值与该节点的左右子节点的最大贡献值
                int priceNewpath = node.val + leftGain + rightGain;

                // 更新答案
                maxSum = Math.max(maxSum, priceNewpath);

                // 返回结点的最大贡献值
                return node.val + Math.max(leftGain, rightGain);
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)

## 51. 岛屿数量
```
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

示例 2：
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```
- 题解
    - 深度优先搜索
        ```
        class Solution {
            void dfs(char[][] grid, int r, int c) {
                int nr = grid.length;
                int nc = grid[0].length;

                if (r < 0 || c <0 || r >= nr || c >= nc || grid[r][c] == '0') {
                    return;
                }

                grid[r][c] = '0';
                dfs(grid, r - 1, c);
                dfs(grid, r + 1, c);
                dfs(grid, r, c - 1);
                dfs(grid, r, c + 1);
            }

            public int numIslands(char[][] grid) {
                if (grid == null || grid.length == 0) {
                    return 0;
                }

                int nr = grid.length;
                int nc = grid[0].length;
                int num_islands = 0;
                for (int r = 0; r < nr; ++r) {
                    for (int c = 0; c < nc; ++c) {
                        if (grid[r][c] == '1') {
                            ++num_islands;
                            dfs(grid, r, c);
                        }
                    }
                }
                
                return num_islands;
            }
        }
        ```
        - 时间复杂度：O(MN)  
        - 空间复杂度：O(MN)
    - 广度优先搜索
        ```
        class Solution {
            public int numIslands(char[][] grid) {
                if (grid == null || grid.length == 0) {
                    return 0;
                }

                int nr = grid.length;
                int nc = grid[0].length;
                int num_islands = 0;
                for (int r = 0; r < nr; ++r) {
                    for (int c = 0; c < nc; ++c) {
                        if (grid[r][c] == '1') {
                            ++num_islands;
                            grid[r][c] = '0';
                            Queue<Integer> neighbors = new LinkedList<>();
                            neighbors.add(r * nc + c);
                            while (!neighbors.isEmpty()) {
                                int id = neighbors.remove();
                                int row = id / nc;
                                int col = id % nc;
                                if (row - 1 >= 0 && grid[row - 1][col] == '1') {
                                    neighbors.add((row - 1) * nc + col);
                                    grid[row - 1][col] = '0';
                                }
                                if (row + 1 < nr && grid[row + 1][col] == '1') {
                                    neighbors.add((row + 1) * nc + col);
                                    grid[row + 1][col] = '0';
                                }
                                if (col - 1 >= 0 && grid[row][col - 1] == '1') {
                                    neighbors.add(row * nc + col - 1);
                                    grid[row][col - 1] = '0';
                                }
                                if (col + 1 < nc && grid[row][col + 1] == '1') {
                                    neighbors.add(row * nc + col + 1);
                                    grid[row][col + 1] = '0';
                                }
                            }
                        }
                    }
                }
                
                return num_islands;
            }
        }
        ```
        - 时间复杂度：O(MN)  
        - 空间复杂度：O(min(M,N))

## 52. 腐烂的橘子
```
在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：

值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。

返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。

示例 1：
输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
输出：4

示例 2：
输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个方向上。

示例 3：
输入：grid = [[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
```
![腐烂的橘子-示例1](/相关图册/腐烂的橘子-示例1.png)     
- 题解
    - 多源广度优先搜索
        ```
        class Solution {
            int[] dr = new int[]{-1, 0, 1, 0};
            int[] dc = new int[]{0, -1, 0, 1};

            public int orangesRotting(int[][] grid) {
                int R = grid.length, C = grid[0].length;
                Queue<Integer> queue = new ArrayDeque<Integer>();
                Map<Integer, Integer> depth  = new HashMap<Integer, Integer>();
                for (int r = 0; r < R; ++r) {
                    for (int c = 0; c < C; ++c) {
                        if (grid[r][c] == 2) {
                            int code = r * C + c;
                            queue.add(code);
                            depth.put(code, 0);
                        }
                    }
                }
                int ans = 0;
                while (!queue.isEmpty()) {
                    int code = queue.remove();
                    int r = code / C, c = code % C;
                    for (int k = 0; k < 4; ++k) {
                        int nr = r + dr[k];
                        int nc = c + dc[k];
                        if (0 <= nr && nr < R && 0 <= nc && nc < C && grid[nr][nc] == 1) {
                            grid[nr][nc] = 2;
                            int ncode = nr * C + nc;
                            queue.add(ncode);
                            depth.put(ncode, depth.get(code) + 1);
                            ans = depth.get(ncode);
                        }
                    }
                } 
                for (int [] row: grid) {
                    for (int v: row) {
                        if (v == 1) {
                            return -1;
                        }
                    }
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(nm)  
        - 空间复杂度：O(nm)

## 53. 课程表
```
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

示例 1：
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。

示例 2：
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成​课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
```
- 题解
    - 深度优先搜索
        ```
        class Solution {
            List<List<Integer>> edges;
            int[] visited;
            boolean valid = true;

            public boolean canFinish(int numCourses, int[][] prerequisites) {
                edges = new ArrayList<List<Integer>>();
                for (int i = 0; i < numCourses; ++i) {
                    edges.add(new ArrayList<Integer>());
                }
                visited = new int[numCourses];
                for (int[] info : prerequisites) {
                    edges.get(info[1]).add(info[0]);
                }
                for (int i = 0; i < numCourses && valid; ++i) {
                    if (visited[i] == 0) {
                        dfs(i);
                    }
                }
                return valid;
            }

            public void dfs(int u) {
                visited[u] = 1;
                for (int v : edges.get(u)) {
                    if (visited[v] == 0) {
                        dfs(v);
                        if (!valid) {
                            return;
                        }
                    } else if (visited[v] == 1) {
                        valid = false;
                        return;
                    }
                }
                visited[u] = 2;
            }
        }
        ```
        - 时间复杂度：O(n+m)  
        - 空间复杂度：O(n+m)
    - 广度优先搜索
        ```
        class Solution {
            List<List<Integer>> edges;
            int[] indeg;

            public boolean canFinish(int numCourses, int[][] prerequisites) {
                edges = new ArrayList<List<Integer>>();
                for (int i = 0; i < numCourses; ++i) {
                    edges.add(new ArrayList<Integer>());
                }
                indeg = new int[numCourses];
                for (int[] info : prerequisites) {
                    edges.get(info[1]).add(info[0]);
                    ++indeg[info[0]];
                }

                Queue<Integer> queue = new LinkedList<Integer>();
                for (int i = 0; i < numCourses; ++i) {
                    if (indeg[i] == 0) {
                        queue.offer(i);
                    }
                }

                int visited = 0;
                while (!queue.isEmpty()) {
                    ++visited;
                    int u = queue.poll();
                    for (int v : edges.get(u)) {
                        --indeg[v];
                        if (indeg[v] == 0) {
                            queue.offer(v);
                        }
                    }
                }

                return visited == numCourses;
            }
        }
        ```
        - 时间复杂度：O(n+m)  
        - 空间复杂度：O(n+m)

## 54. 实现 Trie (前缀树)
```
Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
 
示例：
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]
解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```
- 题解
    - 字典树
        ```
        class Trie {
            private Trie[] children;
            private boolean isEnd;

            public Trie() {
                children = new Trie[26];
                isEnd = false;
            }
            
            public void insert(String word) {
                Trie node = this;
                for (int i = 0; i < word.length(); i++) {
                    char ch = word.charAt(i);
                    int index = ch - 'a';
                    if (node.children[index] == null) {
                        node.children[index] = new Trie();
                    }
                    node = node.children[index];
                }
                node.isEnd = true;
            }
            
            public boolean search(String word) {
                Trie node = searchPrefix(word);
                return node != null && node.isEnd;
            }
            
            public boolean startsWith(String prefix) {
                return searchPrefix(prefix) != null;
            }

            private Trie searchPrefix(String prefix) {
                Trie node = this;
                for (int i = 0; i < prefix.length(); i++) {
                    char ch = prefix.charAt(i);
                    int index = ch - 'a';
                    if (node.children[index] == null) {
                        return null;
                    }
                    node = node.children[index];
                }
                return node;
            }
        }

        /**
        * Your Trie object will be instantiated and called as such:
        * Trie obj = new Trie();
        * obj.insert(word);
        * boolean param_2 = obj.search(word);
        * boolean param_3 = obj.startsWith(prefix);
        */
        ```
        - 时间复杂度：初始化为 O(1)，其余操作为 O(∣S∣)，其中 ∣S∣ 是每次插入或查询的字符串的长度。
        - 空间复杂度：O(∣T∣⋅Σ)，其中 ∣T∣ 为所有插入字符串的长度之和，Σ 为字符集的大小，本题 Σ=26。

## 55. 全排列
```
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

示例 1：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

示例 2：
输入：nums = [0,1]
输出：[[0,1],[1,0]]

示例 3：
输入：nums = [1]
输出：[[1]]
```
- 题解
    - 回溯
        ```
        class Solution {
            public List<List<Integer>> permute(int[] nums) {
                List<List<Integer>> res = new ArrayList<List<Integer>>();

                List<Integer> output = new ArrayList<Integer>();
                for (int num : nums) {
                    output.add(num);
                }

                int n = nums.length;
                backtrack(n, output, res, 0);
                return res;
            }

            public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
                // 所有数都填完了
                if (first == n) {
                    res.add(new ArrayList<Integer>(output));
                } 
                for (int i = first; i < n; i++) {
                    // 动态维护数组
                    Collections.swap(output, first, i);
                    // 继续递归填下一个数
                    backtrack(n, output, res, first + 1);
                    // 撤销操作
                    Collections.swap(output, first, i);
                }
            }
        }
        ```
        - 时间复杂度：O(n×n!)  
        - 空间复杂度：O(n)

## 56. 子集
```
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

示例 1：
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

示例 2：
输入：nums = [0]
输出：[[],[0]]
```
- 题解
    - 迭代法实现子集枚举
        ```
        class Solution {
            List<Integer> t = new ArrayList<Integer>();
            List<List<Integer>> ans = new ArrayList<List<Integer>>();

            public List<List<Integer>> subsets(int[] nums) {
                int n = nums.length;
                for (int mask = 0; mask < (1 << n); ++mask) {
                    t.clear();
                    for (int i = 0; i < n; ++i) {
                        if ((mask & (1 << i)) != 0) {
                            t.add(nums[i]);
                        }
                    }
                    ans.add(new ArrayList<Integer>(t));
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n×2^n)  
        - 空间复杂度：O(n)
    - 递归法实现子集枚举
        ```
        class Solution {
            List<Integer> t = new ArrayList<Integer>();
            List<List<Integer>> ans = new ArrayList<List<Integer>>();

            public List<List<Integer>> subsets(int[] nums) {
                dfs(0, nums);
                return ans;
            }

            public void dfs(int cur, int[] nums) {
                if (cur == nums.length) {
                    ans.add(new ArrayList<Integer>(t));
                    return;
                }
                t.add(nums[cur]);
                dfs(cur + 1, nums);
                t.remove(t.size() - 1);
                dfs(cur + 1, nums);
            }
        }
        ```
        - 时间复杂度：O(n×2^n)  
        - 空间复杂度：O(n)

## 57. 电话号码的字母组合
![电话号码的字母组合](/相关图册/电话号码的字母组合.png)    
```
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

示例 1：
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

示例 2：
输入：digits = ""
输出：[]

示例 3：
输入：digits = "2"
输出：["a","b","c"]
```
- 题解
    - 回溯
        ```
        class Solution {
            public List<String> letterCombinations(String digits) {
                List<String> combinations = new ArrayList<String>();
                if (digits.length() == 0) {
                    return combinations;
                }
                Map<Character, String> phoneMap = new HashMap<Character, String>() {{
                    put('2', "abc");
                    put('3', "def");
                    put('4', "ghi");
                    put('5', "jkl");
                    put('6', "mno");
                    put('7', "pqrs");
                    put('8', "tuv");
                    put('9', "wxyz");
                }};
                backtrack(combinations, phoneMap, digits, 0, new StringBuffer());
                return combinations;
            }

            public void backtrack(List<String> combinations, Map<Character, String> phoneMap, String digits, int index, StringBuffer combination) {
                if (index == digits.length()) {
                combinations.add(combination.toString());
                } else {
                    char digit = digits.charAt(index);
                    String letters = phoneMap.get(digit);
                    int lettersCount = letters.length();
                    for (int i = 0; i < lettersCount; i++) {
                        combination.append(letters.charAt(i));
                        backtrack(combinations, phoneMap, digits, index + 1, combination);
                        combination.deleteCharAt(index);
                    }
                }
            }
        }
        ```
        - 时间复杂度：O(3^m×4^n)  
        - 空间复杂度：O(m+n)

## 58. 组合总和
```
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

示例 1：
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。

示例 2：
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]

示例 3：
输入: candidates = [2], target = 1
输出: []
```
- 题解
    - 搜索回溯
        ```
        class Solution {
            public List<List<Integer>> combinationSum(int[] candidates, int target) {
                List<List<Integer>> ans = new ArrayList<List<Integer>>();
                List<Integer> combine = new ArrayList<Integer>();
                dfs(candidates, target, ans, combine, 0);
                return ans;
            }

            public void dfs(int[] candidates,int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
                if (idx == candidates.length) {
                    return;
                }
                if (target == 0) {
                    ans.add(new ArrayList<Integer>(combine));
                    return;
                }
                // 直接跳过
                dfs(candidates, target, ans, combine, idx + 1);
                // 选择当前数
                if (target - candidates[idx] >= 0) {
                    combine.add(candidates[idx]);
                    dfs(candidates, target - candidates[idx], ans, combine, idx);
                    combine.remove(combine.size() - 1);
                }
            }
        }
        ```
    - 剪枝回溯
        ```
        class Solution {
            public List<List<Integer>> combinationSum(int[] candidates, int target) {
                List<List<Integer>> ans = new ArrayList<List<Integer>>();
                List<Integer> combine = new ArrayList<Integer>();
                dfs(candidates, target, ans, combine, 0);
                return ans;
            }

            public void dfs(int[] candidates,int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
                if (idx == candidates.length) {
                    return;
                }
                if (target == 0) {
                    ans.add(new ArrayList<Integer>(combine));
                    return;
                }
                // 直接跳过
                dfs(candidates, target, ans, combine, idx + 1);
                // 选择当前数
                if (target - candidates[idx] >= 0) {
                    combine.add(candidates[idx]);
                    dfs(candidates, target - candidates[idx], ans, combine, idx);
                    combine.remove(combine.size() - 1);
                }
            }
        }
        ```

## 59. 括号生成
```
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

示例 1：
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

示例 2：
输入：n = 1
输出：["()"]
```
- 题解
    - 暴力法
        ```
        class Solution {
            public List<String> generateParenthesis(int n) {
                List<String> combinations = new ArrayList<String>();
                generateAll(new char[2 * n], 0, combinations);
                return combinations;
            }

            public void generateAll(char[] current, int pos, List<String> result) {
                if (pos == current.length) {
                    if (valid(current)) {
                        result.add(new String(current));
                    }
                } else {
                    current[pos] = '(';
                    generateAll(current, pos + 1, result);
                    current[pos] = ')';
                    generateAll(current, pos + 1, result);
                }
            }

            public boolean valid(char[] current) {
                int balance = 0;
                for (char c : current) {
                    if (c == '(') {
                        ++balance;
                    } else {
                        --balance;
                    }
                    if (balance < 0) {
                        return false;
                    }
                }
                return balance == 0;
            }
        }
        ```
        - 时间复杂度：O(3^m×4^n)  
        - 空间复杂度：O(m+n)
    - 回溯法
        ```
        class Solution {
            public List<String> generateParenthesis(int n) {
                List<String> ans = new ArrayList<String>();
                backtrack(ans, new StringBuilder(), 0, 0, n);
                return ans;
            }

            public void backtrack(List<String> ans, StringBuilder cur, int open, int close, int max) {
                if (cur.length() == max * 2) {
                    ans.add(cur.toString());
                    return;
                }
                if (open < max) {
                    cur.append('(');
                    backtrack(ans, cur, open + 1, close, max);
                    cur.deleteCharAt(cur.length() - 1);
                }
                if (close < open) {
                    cur.append(')');
                    backtrack(ans, cur, open, close + 1, max);
                    cur.deleteCharAt(cur.length() - 1);
                }
            }
        }
        ```
        - 时间复杂度：O((4^n)/√n)  
        - 空间复杂度：O(n)

## 60. 单词搜索
```
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

示例 1：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

示例 2：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true

示例 3：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```
![单词搜索-示例1](/相关图册/单词搜索-示例1.jpg)         
![单词搜索-示例2](/相关图册/单词搜索-示例2.jpg)     
![单词搜索-示例3](/相关图册/单词搜索-示例3.jpg)     
- 题解
    - 回溯
        ```
        class Solution {
            public boolean exist(char[][] board, String word) {
                int h = board.length, w = board[0].length;
                boolean[][] visited = new boolean[h][w];
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        boolean flag = check(board, visited, i, j, word, 0);
                        if (flag) {
                            return true;
                        }
                    }
                }
                return false;
            }

            public boolean check(char[][] board, boolean[][] visited, int i, int j, String s, int k) {
                if (board[i][j] != s.charAt(k)) {
                    return false;
                } else if (k == s.length() - 1) {
                    return true;
                }
                visited[i][j] = true;
                int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
                boolean result = false;
                for (int[] dir : directions) {
                    int newi = i + dir[0], newj = j + dir[1];
                    if (newi >= 0 && newi < board.length && newj >= 0 && newj < board[0].length) {
                        if (!visited[newi][newj]) {
                            boolean flag = check(board, visited, newi, newj, s, k + 1);
                            if (flag) {
                                result = true;
                                break;
                            }
                        }
                    }
                }
                visited[i][j] = false;
                return result;
            }
        }
        ```
        - 时间复杂度：O(MN⋅3^L)  
        - 空间复杂度：O(MN)

## 61. 分割回文串
```
给你一个字符串 s，请你将 s 分割成一些 子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

示例 1：
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]

示例 2：
输入：s = "a"
输出：[["a"]]
```
- 题解
    - 回溯 + 动态规划预处理
        ```
        class Solution {
            boolean[][] f;
            List<List<String>> ret = new ArrayList<List<String>>();
            List<String> ans = new ArrayList<String>();
            int n;

            public List<List<String>> partition(String s) {
                n = s.length();
                f = new boolean[n][n];
                for (int i = 0; i < n; ++i) {
                    Arrays.fill(f[i], true);
                }

                for (int i = n - 1; i >= 0; --i) {
                    for (int j = i + 1; j < n; ++j) {
                        f[i][j] = (s.charAt(i) == s.charAt(j)) && f[i + 1][j - 1];
                    }
                }

                dfs(s, 0);
                return ret;
            }

            public void dfs(String s, int i) {
                if (i == n) {
                    ret.add(new ArrayList<String>(ans));
                    return;
                }
                for (int j = i; j < n; ++j) {
                    if (f[i][j]) {
                        ans.add(s.substring(i, j + 1));
                        dfs(s, j + 1);
                        ans.remove(ans.size() - 1);
                    }
                }
            }
        }
        ```
        - 时间复杂度：O(n⋅2^n)  
        - 空间复杂度：O(n²)
    - 回溯 + 记忆化搜索
        ```
        class Solution {
            int[][] f;
            List<List<String>> ret = new ArrayList<List<String>>();
            List<String> ans = new ArrayList<String>();
            int n;

            public List<List<String>> partition(String s) {
                n = s.length();
                f = new int[n][n];

                dfs(s, 0);
                return ret;
            }

            public void dfs(String s, int i) {
                if (i == n) {
                    ret.add(new ArrayList<String>(ans));
                    return;
                }
                for (int j = i; j < n; ++j) {
                    if (isPalindrome(s, i, j) == 1) {
                        ans.add(s.substring(i, j + 1));
                        dfs(s, j + 1);
                        ans.remove(ans.size() - 1);
                    }
                }
            }

            // 记忆化搜索中，f[i][j] = 0 表示为搜索，1 表示是回文串，-1表示不是回文串
            public int isPalindrome(String s, int i, int j) {
                if (f[i][j] != 0) {
                    return f[i][j];
                }
                if (i >= j) {
                    f[i][j] = 1;
                } else if (s.charAt(i) == s.charAt(j)) {
                    f[i][j] = isPalindrome(s, i + 1, j - 1);
                } else {
                    f[i][j] = -1;
                }
                return f[i][j];
            }
        }
        ```
        - 时间复杂度：O(n⋅2^n)  
        - 空间复杂度：O(n²)

## 62. N 皇后
```
按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

示例 1：
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。

示例 2：
输入：n = 1
输出：[["Q"]]
```
![N 皇后-示例1](/相关图册/N%20皇后-示例1.jpg)  
- 题解
    - 基于集合的回溯
        ```
        class Solution {
            public List<List<String>> solveNQueens(int n) {
                List<List<String>> solutions = new ArrayList<List<String>>();
                int[] queens = new int[n];
                Arrays.fill(queens, -1);
                Set<Integer> columns = new HashSet<Integer>();
                Set<Integer> diagonals1 = new HashSet<Integer>();
                Set<Integer> diagonals2 = new HashSet<Integer>();
                backtrack(solutions, queens, n, 0, columns, diagonals1, diagonals2);
                return solutions;
            }

            public void backtrack(List<List<String>> solutions, int[] queens, int n, int row, Set<Integer> columns, Set<Integer> diagonals1, Set<Integer> diagonals2) {
                if (row == n) {
                    List<String> board = generateBoard(queens, n);
                    solutions.add(board);
                } else {
                    for (int i = 0; i < n; i++) {
                        if (columns.contains(i)) {
                            continue;
                        }
                        int diagonal1 = row - i;
                        if (diagonals1.contains(diagonal1)) {
                            continue;
                        }
                        int diagonal2 = row + i;
                        if (diagonals2.contains(diagonal2)) {
                            continue;
                        }
                        queens[row] = i;
                        columns.add(i);
                        diagonals1.add(diagonal1);
                        diagonals2.add(diagonal2);
                        backtrack(solutions, queens, n, row + 1, columns, diagonals1, diagonals2);
                        queens[row] = -1;
                        columns.remove(i);
                        diagonals1.remove(diagonal1);
                        diagonals2.remove(diagonal2);
                    }
                }
            }

            public List<String> generateBoard(int[] queens, int n) {
                List<String> board = new ArrayList<String>();
                for (int i = 0; i < n; i++) {
                    char[] row = new char[n];
                    Arrays.fill(row, '.');
                    row[queens[i]] = 'Q';
                    board.add(new String(row));
                }
                return board;
            }
        }
        ```
        - 时间复杂度：O(N!)  
        - 空间复杂度：O(N)

## 63. 搜索插入位置
```
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

示例 1:
输入: nums = [1,3,5,6], target = 5
输出: 2

示例 2:
输入: nums = [1,3,5,6], target = 2
输出: 1

示例 3:
输入: nums = [1,3,5,6], target = 7
输出: 4
```
- 题解
    - 二分查找
        ```
        class Solution {
            public int searchInsert(int[] nums, int target) {
                int n = nums.length;
                int left = 0, right = n - 1;
                while (left <= right) {
                    int mid = ((right - left) >> 1) + left;
                    if (target <= nums[mid]) {
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                }
                return left;
            }
        }
        ```
        - 时间复杂度：O(logn)  
        - 空间复杂度：O(1)

## 64. 搜索二维矩阵
```
给你一个满足下述两条属性的 m x n 整数矩阵：

每行中的整数从左到右按非严格递增顺序排列。
每行的第一个整数大于前一行的最后一个整数。
给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。

示例 1：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true

示例 2：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false
```
![搜索二维矩阵-示例1](/相关图册/搜索二维矩阵-示例1.jpg)  
![搜索二维矩阵-示例2](/相关图册/搜索二维矩阵-示例2.jpg)
- 题解
    - 两次二分查找
        ```
        class Solution {
            public boolean searchMatrix(int[][] matrix, int target) {
                int rowIndex = binarySearchFirstColumn(matrix, target);
                if (rowIndex < 0) {
                    return false;
                }
                return binarySearchRow(matrix[rowIndex], target);
            }

            public int binarySearchFirstColumn(int[][] matrix, int target) {
                int low = -1, high = matrix.length - 1;
                while (low < high) {
                    int mid = (high - low + 1) / 2 + low;
                    if (matrix[mid][0] <= target) {
                        low = mid;
                    } else {
                        high = mid - 1;
                    }
                }
                return low;
            }

            public boolean binarySearchRow(int[] row, int target) {
                int low = 0, high = row.length - 1;
                while (low <= high) {
                    int mid = (high - low) / 2 + low;
                    if (row[mid] == target) {
                        return true;
                    } else if (row[mid] > target) {
                        high = mid - 1;
                    } else {
                        low = mid + 1;
                    }
                }
                return false;
            }
        }
        ```
        - 时间复杂度：O(logmn)  
        - 空间复杂度：O(1)
    - 一次二分查找
        ```
        class Solution {
            public boolean searchMatrix(int[][] matrix, int target) {
                int m = matrix.length, n = matrix[0].length;
                int low = 0, high = m * n - 1;
                while (low <= high) {
                    int mid = (high - low) / 2 + low;
                    int x = matrix[mid / n][mid % n];
                    if (x < target) {
                        low = mid + 1;
                    } else if (x > target) {
                        high = mid - 1;
                    } else {
                        return true;
                    }
                }
                return false;
            }
        }
        ```   
        - 时间复杂度：O(logmn)  
        - 空间复杂度：O(1)

## 65. 在排序数组中查找元素的第一个和最后一个位置 
```
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

示例 1：
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

示例 2：
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

示例 3：
输入：nums = [], target = 0
输出：[-1,-1]
```
- 题解
    - 二分查找
        ```
        class Solution {
            public int[] searchRange(int[] nums, int target) {
                int leftIdx = binarySearch(nums, target, true);
                int rightIdx = binarySearch(nums, target, false) - 1;
                if (leftIdx <= rightIdx && rightIdx < nums.length && nums[leftIdx] == target && nums[rightIdx] == target) {
                    return new int[]{leftIdx, rightIdx};
                }
                return new int[]{-1, -1};
            }

            public int binarySearch(int[] nums, int target, boolean lower) {
                int left = 0, right = nums.length - 1, ans = nums.length;
                while (left <= right) {
                    int mid = (left + right) / 2;
                    if (nums[mid] > target || (lower && nums[mid] >= target)) {
                        right = mid - 1;
                        ans = mid;
                    } else {
                        left = mid + 1;
                    }
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(logn)  
        - 空间复杂度：O(1)

## 66. 搜索旋转排序数组
```
整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例 2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例 3：
输入：nums = [1], target = 0
输出：-1
```
- 题解
    - 二分查找
        ```
        class Solution {
            public int search(int[] nums, int target) {
                int n = nums.length;
                if (n == 0) {
                    return -1;
                }
                if (n == 1) {
                    return nums[0] == target ? 0 : -1;
                }
                int l = 0, r = n - 1;
                while (l <= r) {
                    int mid = (l + r) / 2;
                    if (nums[mid] == target) {
                        return mid;
                    }
                    if (nums[0] <= nums[mid]) {
                        if (nums[0] <= target && target < nums[mid]) {
                            r = mid - 1;
                        } else {
                            l = mid + 1;
                        }
                    }  else {
                        if (nums[mid] < target && target <= nums[n - 1]) {
                            l = mid + 1;
                        } else {
                            r = mid - 1;
                        }
                    }
                }
                return -1;
            }
        }
        ```
        - 时间复杂度：O(logn)  
        - 空间复杂度：O(1)

## 67. 寻找旋转排序数组中的最小值
```
已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

示例 1：
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。

示例 2：
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。

示例 3：
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
```
- 题解
    - 二分查找
        ```
        class Solution {
            public int findMin(int[] nums) {
                int low = 0;
                int high = nums.length - 1;
                while (low < high) {
                    int pivot = low + (high - low) / 2;
                    if (nums[pivot] < nums[high]) {
                        high = pivot;
                    } else {
                        low = pivot + 1;
                    }
                }
                return nums[low];
            }
        }
        ```
        - 时间复杂度：O(logn)  
        - 空间复杂度：O(1)

## 68. 寻找两个正序数组的中位数
```
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

示例 1：
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

示例 2：
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```
- 题解
    - 归并
        ```
        class Solution {
            public double findMedianSortedArrays(int[] nums1, int[] nums2) {
                int nums1Size = nums1.length;
                int nums2Size = nums2.length;
                int size = nums1Size + nums2Size;
                int[] num = new int[size];
                int i = 0, j = 0, k = 0;
                while(i < nums1Size && j < nums2Size) {
                    if (nums1[i] <= nums2[j]) {
                        num[k++] = nums1[i++];
                    } else {
                        num[k++] = nums2[j++];
                    }
                }
                while (i < nums1Size) {
                    num[k++] = nums1[i++];
                }
                while (j < nums2Size) {
                    num[k++] = nums2[j++];
                }
                return (num[size / 2] + num[(size - 1) / 2]) / 2.0;
            }
        }
        ```
        - 时间复杂度：O(m+n)  
        - 空间复杂度：O(m+n)

## 69. 有效的括号
```
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。

示例 1：
输入：s = "()"
输出：true

示例 2：
输入：s = "()[]{}"
输出：true

示例 3：
输入：s = "(]"
输出：false

示例 4：
输入：s = "([])"
输出：true
```
- 题解
    - 栈
        ```
        class Solution {
            public boolean isValid(String s) {
                int n = s.length();
                if (n % 2 == 1) {
                    return false;
                }

                Map<Character, Character> pairs = new HashMap<Character, Character>() {{
                    put(')', '(');
                    put(']', '[');
                    put('}', '{');
                }};
                Deque<Character> stack = new LinkedList<Character>();
                for (int i = 0; i < n; i++) {
                    char ch = s.charAt(i);
                    if (pairs.containsKey(ch)) {
                        if (stack.isEmpty() || stack.peek() != pairs.get(ch)) {
                            return false;
                        }
                        stack.pop();
                    } else {
                        stack.push(ch);
                    }
                }
                return stack.isEmpty();
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 70. 最小栈
```
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。

示例 1:
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
输出：
[null,null,null,null,-3,null,0,-2]
解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```
- 题解
    - 辅助栈
        ```
        class MinStack {
            Deque<Integer> xStack;
            Deque<Integer> minStack;

            public MinStack() {
                xStack = new LinkedList<Integer>();
                minStack = new LinkedList<Integer>();
                minStack.push(Integer.MAX_VALUE);
            }
            
            public void push(int val) {
                xStack.push(val);
                minStack.push(Math.min(minStack.peek(), val));
            }
            
            public void pop() {
                xStack.pop();
                minStack.pop();        
            }
            
            public int top() {
                return xStack.peek();
            }
            
            public int getMin() {
                return minStack.peek();
            }
        }

        /**
        * Your MinStack object will be instantiated and called as such:
        * MinStack obj = new MinStack();
        * obj.push(val);
        * obj.pop();
        * int param_3 = obj.top();
        * int param_4 = obj.getMin();
        */
        ```
        - 时间复杂度：O(1)  
        - 空间复杂度：O(n)

## 71. 字符串解码
```
给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

示例 1：
输入：s = "3[a]2[bc]"
输出："aaabcbc"

示例 2：
输入：s = "3[a2[c]]"
输出："accaccacc"

示例 3：
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"

示例 4：
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```
- 题解
    - 栈操作
        ```
        class Solution {
            int ptr;

            public String decodeString(String s) {
                LinkedList<String> stk = new LinkedList<String>();
                ptr = 0;

                while (ptr < s.length()) {
                    char cur = s.charAt(ptr);
                    if (Character.isDigit(cur)) {
                        // 获取一个数字并进栈
                        String digits = getDigits(s);
                        stk.addLast(digits);
                    } else if (Character.isLetter(cur) || cur == '[') {
                        // 获取一个字母并进栈
                        stk.addLast(String.valueOf(s.charAt(ptr++)));
                    } else {
                        ++ptr;
                        LinkedList<String> sub = new LinkedList<String>();
                        while (!"[".equals(stk.peekLast())) {
                            sub.addLast(stk.removeLast());
                        }
                        Collections.reverse(sub);
                        // 左括号出栈
                        stk.removeLast();
                        // 此时栈顶为当前 sub 对应的字符串应该出现的次数
                        int repTime = Integer.parseInt(stk.removeLast());
                        StringBuffer t = new StringBuffer();
                        String o = getString(sub);
                        // 构造字符串
                        while (repTime-- > 0) {
                            t.append(o);
                        }
                        // 将构造好的字符串入栈
                        stk.addLast(t.toString());
                    }
                }

                return getString(stk);
            }

            public String getDigits(String s) {
                StringBuffer ret = new StringBuffer();
                while (Character.isDigit(s.charAt(ptr))) {
                    ret.append(s.charAt(ptr++));
                }
                return ret.toString();
            }

            public String getString(LinkedList<String> v) {
                StringBuffer ret = new StringBuffer();
                for (String s : v) {
                    ret.append(s);
                }
                return ret.toString();
            }
        }
        ```
        - 时间复杂度：O(S)  
        - 空间复杂度：O(S)

## 72. 每日温度
```
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

示例 1:
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]

示例 2:
输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]

示例 3:
输入: temperatures = [30,60,90]
输出: [1,1,0]
```
- 题解
    - 单调栈
        ```
        class Solution {
            public int[] dailyTemperatures(int[] temperatures) {
                int length = temperatures.length;
                int[] ans = new int[length];
                Deque<Integer> stack = new LinkedList<Integer>();
                for (int i = 0; i < length; i++) {
                    int temperature = temperatures[i];
                    while (!stack.isEmpty() && temperature > temperatures[stack.peek()]) {
                        int prevIndex = stack.pop();
                        ans[prevIndex] = i - prevIndex;
                    }
                    stack.push(i);
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)

## 73. 柱状图中最大的矩形
```
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

示例 1:
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10

示例 2：
输入： heights = [2,4]
输出： 4
```
![柱状图中最大的矩形-示例1](/相关图册/柱状图中最大的矩形-示例1.jpg)  
![柱状图中最大的矩形-示例2](/相关图册/柱状图中最大的矩形-示例2.jpg)
- 题解
    - 单调栈
        ```
        class Solution {
            public int largestRectangleArea(int[] heights) {
                int n = heights.length;
                int[] left = new int[n];
                int[] right = new int[n];

                Deque<Integer> mono_stack = new ArrayDeque<Integer>();
                for (int i = 0; i < n; ++i) {
                    while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                        mono_stack.pop();
                    }
                    left[i] = (mono_stack.isEmpty() ? -1 : mono_stack.peek());
                    mono_stack.push(i);
                }

                mono_stack.clear();
                for (int i = n - 1; i >= 0; --i) {
                    while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                        mono_stack.pop();
                    }
                    right[i] = (mono_stack.isEmpty() ? n : mono_stack.peek());
                    mono_stack.push(i);
                }

                int ans = 0;
                for (int i = 0; i < n; ++i) {
                    ans = Math.max(ans, (right[i] - left[i] - 1) * heights[i]);
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(N)  
        - 空间复杂度：O(N)

## 74. 数组中的第K个最大元素
```
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例 1:
输入: [3,2,1,5,6,4], k = 2
输出: 5

示例 2:
输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```
- 题解
    - 基于堆排序的选择方法
        ```
        class Solution {
            public int findKthLargest(int[] nums, int k) {
                int heapSize = nums.length;
                buildMaxheap(nums, heapSize);
                for (int i = nums.length - 1; i >=nums.length - k + 1; --i) {
                    swap(nums, 0, i);
                    --heapSize;
                    maxHeapify(nums, 0, heapSize);
                }
                return nums[0];
            }

            public void buildMaxheap(int[] a, int heapSize) {
                for (int i = heapSize / 2 - 1; i >= 0; --i) {
                    maxHeapify(a, i, heapSize);
                }
            }

            public void maxHeapify(int[] a, int i, int headSize) {
                int l = i * 2 + 1, r = i * 2 + 2, largerst = i;
                if (l < headSize && a[l] > a[largerst]) {
                    largerst = l;
                }
                if (r < headSize && a[r] > a[largerst]) {
                    largerst = r;
                }
                if (largerst != i) {
                    swap(a, i, largerst);
                    maxHeapify(a, largerst, headSize);
                }
            }

            public void swap(int[] a, int i, int j) {
                int temp = a[i];
                a[i] = a[j];
                a[j] = temp;
            }
        }
        ```
        - 时间复杂度：O(nlogn)  
        - 空间复杂度：O(logn)

## 75. 前 K 个高频元素
```
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

示例 1:
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

示例 2:
输入: nums = [1], k = 1
输出: [1]
```
- 题解
    - 堆
        ```
        class Solution {
            public int[] topKFrequent(int[] nums, int k) {
                Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
                for (int num : nums) {
                    occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
                }

                // int[] 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
                PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
                    public int compare(int[] m, int[] n) {
                        return m[1] - n[1];
                    }
                });
                for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
                    int num = entry.getKey(), count = entry.getValue();
                    if (queue.size() == k) {
                        if (queue.peek()[1] < count) {
                            queue.poll();
                            queue.offer(new int[]{num, count});
                        }
                    } else {
                        queue.offer(new int[]{num, count});
                    }
                } 
                int[] ret = new int[k];
                for (int i = 0; i < k; ++i) {
                    ret[i] = queue.poll()[0];
                }
                return ret; 
            }
        }
        ```
        - 时间复杂度：O(Nlogk)  
        - 空间复杂度：O(N)

## 76. 数据流的中位数
```
中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

例如 arr = [2,3,4] 的中位数是 3 。
例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。

实现 MedianFinder 类:

MedianFinder() 初始化 MedianFinder 对象。
void addNum(int num) 将数据流中的整数 num 添加到数据结构中。
double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。

示例 1：
输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```
- 题解
    - 优先队列
        ```
        class MedianFinder {
            PriorityQueue<Integer> queMin;
            PriorityQueue<Integer> queMax;

            public MedianFinder() {
                queMin = new PriorityQueue<Integer>((a, b) -> (b - a));
                queMax = new PriorityQueue<Integer>((a, b) -> (a - b));
            }
            
            public void addNum(int num) {
                if (queMin.isEmpty() || num <= queMin.peek()) {
                    queMin.offer(num);
                    if (queMax.size() + 1 < queMin.size()) {
                        queMax.offer(queMin.poll());
                    }
                } else {
                    queMax.offer(num);
                    if (queMax.size() > queMin.size()) {
                        queMin.offer(queMax.poll());
                    }
                }
            }
            
            public double findMedian() {
                if (queMin.size() > queMax.size()) {
                    return queMin.peek();
                }
                return (queMin.peek() + queMax.peek()) / 2.0;
            }
        }

        /**
        * Your MedianFinder object will be instantiated and called as such:
        * MedianFinder obj = new MedianFinder();
        * obj.addNum(num);
        * double param_2 = obj.findMedian();
        */
        ```
        - 时间复杂度：O(logn)  
        - 空间复杂度：O(n)

## 77. 买卖股票的最佳时机
```
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

示例 1：
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

示例 2：
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```
- 题解
    - 一次遍历
        ```
        class Solution {
            public int maxProfit(int[] prices) {
                int minprice = Integer.MAX_VALUE;
                int maxprofit = 0;
                for (int i = 0; i < prices.length; i++) {
                    if (prices[i] < minprice) {
                        minprice = prices[i];
                    } else if (prices[i] - minprice > maxprofit) {
                        maxprofit = prices[i] - minprice;
                    }
                }
                return maxprofit;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 78. 跳跃游戏
```
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。

示例 1：
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。

示例 2：
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```
- 题解
    - 贪心
        ```
        class Solution {
            public boolean canJump(int[] nums) {
                int n = nums.length;
                int rightmost = 0;
                for (int i = 0; i < n; ++i) {
                    if (i <= rightmost) {
                        rightmost = Math.max(rightmost, i + nums[i]);
                        if (rightmost >= n - 1) {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 79. 跳跃游戏 II
```
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:

0 <= j <= nums[i] 
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

示例 1:
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

示例 2:
输入: nums = [2,3,0,1,4]
输出: 2
```
- 题解
    - 反向查找出发位置
        ```
        class Solution {
            public int jump(int[] nums) {
                int position = nums.length - 1;
                int steps = 0;
                while (position > 0) {
                    for (int i = 0; i < position; i++) {
                        if (i + nums[i] >= position) {
                            position = i;
                            steps++;
                            break;
                        }
                    }
                }
                return steps;
            }
        }
        ```
        - 时间复杂度：O(n²)  
        - 空间复杂度：O(1)
        
    - 正向查找可到达的最大位置
        ```
        class Solution {
            public int jump(int[] nums) {
                int length = nums.length;
                int end = 0;
                int maxPosition = 0;
                int steps = 0;
                for (int i = 0; i < length - 1; i++) {
                    maxPosition = Math.max(maxPosition, i + nums[i]);
                    if (i == end) {
                        end = maxPosition;
                        steps++;
                    }
                }
                return steps;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 80. 划分字母区间
```
给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。例如，字符串 "ababcc" 能够被分为 ["abab", "cc"]，但类似 ["aba", "bcc"] 或 ["ab", "ab", "cc"] 的划分是非法的。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 s 。

返回一个表示每个字符串片段的长度的列表。

示例 1：
输入：s = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 这样的划分是错误的，因为划分的片段数较少。 

示例 2：
输入：s = "eccbbbbdec"
输出：[10]
```
- 题解
    - 贪心
        ```
        class Solution {
            public List<Integer> partitionLabels(String s) {
                int[] last = new int[26];
                int length = s.length();
                for (int i = 0; i < length; i++) {
                    last[s.charAt(i) - 'a'] = i;
                }
                List<Integer> partition = new ArrayList<Integer>();
                int start = 0, end = 0;
                for (int i = 0; i < length; i++) {
                    end = Math.max(end, last[s.charAt(i) - 'a']);
                    if (i == end) {
                        partition.add(end - start + 1);
                        start = end + 1;
                    }
                }
                return partition;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(∣Σ∣)，其中 Σ 是字符串中的字符集。这道题中，字符串只包含小写字母，因此 ∣Σ∣=26。

## 81. 爬楼梯
```
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

示例 1：
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

示例 2：
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int climbStairs(int n) {
                // 爬一楼
                int p = 1;
                // 爬二楼
                int q = 2;
                if (n == 1) {
                    return p;
                } else if (n == 2) {
                    return q;
                } else {
                    // 从第三楼开始，只有两种上楼方式，从前一层再爬一楼和从前二层再爬两楼。
                    // 可以推出f(n) = f(n - 1) + f(n - 2)
                    // 直接递归会超时，所以用的for循环求结果
                    int r = 0;
                    for (int i = 3; i <= n; i++) {
                        r = q + p;
                        p = q;
                        q = r;
                    }
                    return r;
                }
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 82. 杨辉三角
```
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

示例 1:
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

示例 2:
输入: numRows = 1
输出: [[1]]
```
![杨辉三角-示例1](/相关图册/杨辉三角-示例1.gif)  
- 题解
    - 数学
        ```
        class Solution {
            public List<List<Integer>> generate(int numRows) {
                List<List<Integer>> ret = new ArrayList<List<Integer>>();
                for (int i = 0; i < numRows; ++i) {
                    List<Integer> row = new ArrayList<Integer>();
                    for (int j = 0; j <= i; ++j) {
                        if (j == 0 || j == i) {
                            row.add(1);
                        } else {
                            row.add(ret.get(i - 1).get(j - 1) + ret.get(i - 1).get(j));
                        }
                    }
                    ret.add(row);
                }
                return ret;
            }
        }
        ```
        - 时间复杂度：O(numRows²)  
        - 空间复杂度：O(1)

## 83. 打家劫舍
```
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

示例 1：
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。

示例 2：
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int rob(int[] nums) {
                if (nums == null || nums.length == 0) {
                    return 0;
                }
                int length = nums.length;
                if (length == 1) {
                    return nums[0];
                }
                int[] dp = new int[length];
                dp[0] = nums[0];
                dp[1] = Math.max(nums[0], nums[1]);
                for (int i = 2; i < length; i++) {
                    dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
                }
                return dp[length - 1];
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)
    - 滚动数组
        ```
        class Solution {
            public int rob(int[] nums) {
                if (nums == null || nums.length == 0) {
                    return 0;
                }
                int length = nums.length;
                if (length == 1) {
                    return nums[0];
                }
                int first = nums[0], second = Math.max(nums[0], nums[1]);
                for (int i = 2; i < length; i++) {
                    int temp = second;
                    second = Math.max(first + nums[i], second);
                    first = temp;
                }
                return second;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 84. 完全平方数
```
给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

示例 1：
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4

示例 2：
输入：n = 13
输出：2
解释：13 = 4 + 9
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int numSquares(int n) {
                int f[] = new int[n + 1];
                f[0] = 0;
                for (int i = 1; i <= n; i++) {
                    f[i] = Integer.MAX_VALUE;
                    for (int j  = 1; j * j <= i; j++) {
                        f[i] = Math.min(f[i - j * j] + 1, f[i]);
                    }
                }
                return f[n];
            }
        }
        ```
        - 时间复杂度：O(n√n)  
        - 空间复杂度：O(n)

## 85. 零钱兑换
```
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

示例 1：
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1

示例 2：
输入：coins = [2], amount = 3
输出：-1

示例 3：
输入：coins = [1], amount = 0
输出：0
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int coinChange(int[] coins, int amount) {
                int max = amount + 1;
                int[] dp = new int[amount + 1];
                Arrays.fill(dp, max);
                dp[0] = 0;
                for (int i = 1; i <= amount; i++) {
                    for (int j = 0; j < coins.length; j++) {
                        if (coins[j] <= i) {
                            dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                        }
                    }
                }
                return dp[amount] > amount ? - 1 : dp[amount];
            }
        }
        ```
        - 时间复杂度：O(Sn)，其中 S 是金额，n 是面额数。
        - 空间复杂度：O(S)

## 86. 单词拆分
```
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

示例 1：
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。

示例 2：
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。

示例 3：
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```
- 题解
    - 动态规划
        ```
        class Solution {
            public boolean wordBreak(String s, List<String> wordDict) {
                Set<String> wordDictSet = new HashSet(wordDict);
                boolean[] dp = new boolean[s.length() + 1];
                dp[0] = true;
                for (int i = 1; i <= s.length(); i++) {
                    for (int j = 0; j < i; j++) {
                        if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                            dp[i] = true;
                            break;
                        }
                    }
                }
                return dp[s.length()];
            }
        }
        ```
        - 时间复杂度：O(n²)  
        - 空间复杂度：O(n)

## 87. 最长递增子序列
```
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

示例 1：
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。

示例 2：
输入：nums = [0,1,0,3,2,3]
输出：4

示例 3：
输入：nums = [7,7,7,7,7,7,7]
输出：1
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int lengthOfLIS(int[] nums) {
                if (nums.length == 0) {
                    return 0;
                }
                int[] dp = new int[nums.length];
                dp[0] = 1;
                int maxns = 1;
                for (int i = 1; i < nums.length; i++) {
                    dp[i] = 1;
                    for (int j = 0; j < i; j++) {
                        if (nums[i] > nums[j]) {
                            dp[i] = Math.max(dp[i], dp[j] + 1);
                        }
                    }
                    maxns = Math.max(maxns, dp[i]);
                }
                return maxns;
            }
        }
        ```
        - 时间复杂度：O(n²)  
        - 空间复杂度：O(n)

## 88. 乘积最大子数组
```
给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 32-位 整数。

示例 1:
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

示例 2:
输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int maxProduct(int[] nums) {
                int length = nums.length;
                long[] maxF = new long[length];
                long[] minF = new long[length];
                for (int i = 0; i < length; i++) {
                    maxF[i] = nums[i];
                    minF[i] = nums[i];
                }
                for (int i = 1; i < length; ++i) {
                    maxF[i] = Math.max(maxF[i - 1] * nums[i], Math.max(nums[i], minF[i - 1] * nums[i]));
                    minF[i] = Math.min(minF[i - 1] * nums[i], Math.min(nums[i], maxF[i - 1] * nums[i]));
                    if (minF[i] < (-1 << 31)) {
                        minF[i] = nums[i];
                    }
                }
                int ans = (int) maxF[0];
                for (int i = 1; i < length; ++i) {
                    ans = Math.max(ans, (int) maxF[i]);
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 滚动数组
        ```
        class Solution {
            public int maxProduct(int[] nums) {
                long maxF = nums[0], minF = nums[0];
                int ans = nums[0];
                int length = nums.length;
                for (int i = 1; i < length; ++i) {
                    long mx = maxF, mn = minF;
                    maxF = Math.max(mx * nums[i], Math.max(nums[i], mn * nums[i]));
                    minF = Math.min(mn * nums[i], Math.min(nums[i], mx * nums[i]));
                    if (minF < -1 << 31) {
                        minF = nums[i];
                    }
                    ans = Math.max((int) maxF, ans);
                }
                return ans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)

## 89. 分割等和子集
```
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

示例 1：
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

示例 2：
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```
- 题解
    - 动态规划
        ```
        class Solution {
            public boolean canPartition(int[] nums) {
                int n = nums.length;
                if (n < 2) {
                    return false;
                }
                int sum = 0, maxNum = 0;
                for (int num : nums) {
                    sum += num;
                    maxNum = Math.max(maxNum, num);
                }
                if (sum % 2 != 0) {
                    return false;
                }
                int target = sum / 2;
                if (maxNum > target) {
                    return false;
                }
                boolean[] dp = new boolean[target + 1];
                dp[0] = true;
                for (int i = 0; i < n; i++) {
                    int num = nums[i];
                    for (int j = target; j >= num; --j) {
                        dp[j] |= dp[j - num];
                    }
                }
                return dp[target];
            }
        }
        ```
        - 时间复杂度：O(n × target)  
        - 空间复杂度：O(target)

## 90. 最长有效括号
```
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

示例 1：
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"

示例 2：
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"

示例 3：
输入：s = ""
输出：0
```
- 题解
    - 动态规划
        ```
        class Solution {
            public int longestValidParentheses(String s) {
                int maxans = 0;
                int[] dp = new int[s.length()];
                for (int i = 1; i < s.length(); i++) {
                    if (s.charAt(i) == ')') {
                        if (s.charAt(i - 1) == '(') {
                            dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2; 
                        } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                            dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                        }
                        maxans = Math.max(maxans, dp[i]);
                    }
                }
                return maxans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 栈
        ```
        class Solution {
            public int longestValidParentheses(String s) {
                int maxans = 0;
                Deque<Integer> stack = new LinkedList<Integer>();
                stack.push(-1);
                for (int i = 0; i < s.length(); i++) {
                    if (s.charAt(i) == '(') {
                        stack.push(i);
                    } else {
                        stack.pop();
                        if (stack.isEmpty()) {
                            stack.push(i);
                        } else {
                            maxans = Math.max(maxans, i - stack.peek());
                        }
                    }
                }
                return maxans;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(n)
    - 不需要额外的空间
        ```
        class Solution {
            public int longestValidParentheses(String s) {
                int left = 0, right = 0, maxlength = 0;
                for (int i = 0; i < s.length(); i++) {
                    if (s.charAt(i) == '(') {
                        left++;
                    } else {
                        right++;
                    }
                    if (left == right) {
                        maxlength = Math.max(maxlength, 2 * right);
                    } else if (right > left) {
                        left = right = 0;
                    }
                }
                left = right = 0;
                for (int i = s.length() - 1; i >= 0; i--) {
                    if (s.charAt(i) == '(') {
                        left++;
                    } else {
                        right++;
                    }
                    if (left == right) {
                        maxlength = Math.max(maxlength, 2 * left);
                    } else if (left > right) {
                        left = right = 0;
                    }
                }
                return maxlength;
            }
        }
        ```
        - 时间复杂度：O(n)  
        - 空间复杂度：O(1)







