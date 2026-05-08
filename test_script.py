import re

with open('evo13.py', 'r') as f:
    content = f.read()

# Check for where to put QUALITY_DIVERSITY_ENABLED
print(content.find('EVOLUTION_MODEL = "island"'))
