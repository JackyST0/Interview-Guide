# LeetCode 热题 100

1. 两数之和：
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

2. 字母异位词分组：
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

3. 最长连续序列：
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

4. 移动零：
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

5. 盛最多水的容器：
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

6. 三数之和：
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

7. 接雨水
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

8. 