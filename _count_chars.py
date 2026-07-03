import re

with open(r'D:\claude-code-project\graphRAG\docs\RAG-2\ch10-graphrag-deepseek.md', encoding='utf-8') as f:
    lines = f.readlines()

content = ''.join(lines)
print(f'Total lines: {len(lines)}')

chinese = sum(1 for c in content if '一' <= c <= '鿿')
print(f'Chinese characters: {chinese}')

code_marker = '```'
print(f'Code block markers: {content.count(code_marker)}')

table_marker = '|---|'
print(f'Table headers: {content.count(table_marker)}')

headings = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
print(f'Headings: {len(headings)}')
for h in re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
    print(f'  H{len(h[0])}: {h[1]}')
