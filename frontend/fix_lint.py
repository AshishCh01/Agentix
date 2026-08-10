import os

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove import React from 'react';
    content = content.replace("import React from 'react';\n", "")
    content = content.replace('import React from "react";\n', "")
    
    # Replace import React, { ... } from 'react'; with import { ... } from 'react';
    if 'import React, {' in content:
        content = content.replace('import React, {', 'import {')
    
    # Fix useRef in ChatContext
    if "ChatContext.jsx" in filepath:
        content = content.replace('import { createContext, useState, useEffect, useCallback, useContext } from "react";', 'import { createContext, useState, useEffect, useCallback, useContext, useRef } from "react";')
        
        # Fix setState in effect warning
        old_cleanup = '''    if (!user) {
      setSessions([]);
      setMessages([]);
      setActiveSessionId(null);
      return;
    }'''
        new_cleanup = '''    if (!user) {
      setTimeout(() => {
        setSessions([]);
        setMessages([]);
        setActiveSessionId(null);
      }, 0);
      return;
    }'''
        content = content.replace(old_cleanup, new_cleanup)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

frontend_src = 'src'
for root, _, files in os.walk(frontend_src):
    for f in files:
        if f.endswith('.jsx') or f.endswith('.js'):
            fix_file(os.path.join(root, f))
