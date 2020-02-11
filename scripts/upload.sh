# rsync -azv -e 'ssh -A -J tuandinh@144.92.237.175' src tuandinh@128.104.158.78:~/Documents/Project/milgan/
rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' src/ tuandinh@128.104.158.78:~/Documents/Project/semioptpixel/src/
