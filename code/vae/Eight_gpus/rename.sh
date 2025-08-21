#!/bin/bash

# 要处理的根目录，可按需修改
root_dir="."
# 要替换的旧路径，可按需修改
old_path="/data/code/pp2"
# 新的路径，可按需修改
new_path="/data/code/pp2"


# 遍历所有文件
find "$root_dir" -type f -print0 | while IFS= read -r -d '' file; do
    # 替换文件中的路径
    sed -i "s|$old_path|$new_path|g" "$file"
done

