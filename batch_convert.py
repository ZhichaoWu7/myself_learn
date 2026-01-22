from doc_to_md_converter import DocToMdConverter
import os
from pathlib import Path


def start():
    # 1. 基础路径配置
    # 建议使用绝对路径防止定位出错，这里 /Volumes/soler... 也可以
    base_dir = Path(".").absolute()
    data_dir = base_dir / "data"

    # 2. 关键修改：为了配合 MD 里的相对路径引用 ![img](picture/xxx)
    # 我们把输出目录定为 converted_md，把图片目录定在其子目录 picture 下
    out_dir = base_dir/ "data"
    pix_dir = out_dir / "picture"

    # 如果文件夹不存在，手动创建
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print(f"📁 已创建数据目录 {data_dir}，请放入文件后再运行。")
        return

    # 3. 初始化转换器
    converter = DocToMdConverter(
        out_dir=str(out_dir),
        image_dir=str(pix_dir)
    )

    print(f"开始批量转换并替换... 源目录: {data_dir}")

    # 4. 遍历并执行
    for file in os.listdir(data_dir):
        file_path = data_dir / file

        # 只处理文件，跳过隐藏文件（如 .DS_Store）
        if file_path.is_file() and not file.startswith('.'):
            # 调用你刚才改好的 convert_and_replace
            # 它会：转换 -> 存图 -> 删原文件
            converter.convert_and_replace(str(file_path))

    print("\n" + "=" * 30)
    print("✨ 任务运行完成！")
    print(f"📄 Markdown 存放在: {out_dir}")
    print(f"🖼️ 图片存放在: {pix_dir}")
    print("=" * 30)


if __name__ == "__main__":
    start()