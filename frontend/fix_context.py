def add_eslint_disable(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if '/* eslint-disable react-refresh/only-export-components */' not in content:
        content = '/* eslint-disable react-refresh/only-export-components */\n' + content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

add_eslint_disable('src/context/AuthContext.jsx')
add_eslint_disable('src/context/ChatContext.jsx')
