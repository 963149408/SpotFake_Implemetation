#!/bin/bash
python main.py>$1.txt 2>&1;python mail.py