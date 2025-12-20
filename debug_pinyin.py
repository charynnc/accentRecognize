import os
from pypinyin import pinyin, Style, lazy_pinyin

def test_pinyin_extraction(file_path):
    print(f"--- Testing File: {file_path} ---")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    print(f"Original Text: {text}")
    
    # Current Method (Whole Syllable with Tone)
    pys_tone3 = [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]
    print(f"\nCurrent Feature (TONE3): {pys_tone3}")
    
    # Analysis of Factors (Initials and Finals)
    print("\n--- Factor Analysis (Initials & Finals) ---")
    print("Character | Pinyin (Tone3) | Initial (声母) | Final (韵母) | Tone")
    print("-" * 70)
    
    initials = [p[0] for p in pinyin(text, style=Style.INITIALS, errors='ignore', strict=False)]
    finals = [p[0] for p in pinyin(text, style=Style.FINALS, errors='ignore', strict=False)]
    
    for char, py, ini, fin in zip(text, pys_tone3, initials, finals):
        # Extract tone number if present
        tone = "0"
        if py[-1].isdigit():
            tone = py[-1]
        
        # Handle cases where initial is empty (e.g. 'a', 'an')
        if ini == "":
            ini = "_"
            
        print(f"{char:^9} | {py:^14} | {ini:^12} | {fin:^10} | {tone:^4}")

if __name__ == "__main__":
    # Use the file we found
    sample_file = "/data_extend/fcy/dl/datasets/ST-CMDS-20170001_1-OS/20170001P00162I0038.txt"
    test_pinyin_extraction(sample_file)
