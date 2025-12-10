@echo off
echo Testando animações P2P...
echo.

echo 1. Falha por TTL insuficiente (TTL=2)
.venv\Scripts\python.exe p2p.py config.json animate n1 archive.zip 2 flooding falha_ttl.gif
echo.

echo 2. Sucesso com TTL=3
.venv\Scripts\python.exe p2p.py config.json animate n1 archive.zip 3 flooding sucesso_ttl3.gif
echo.

echo 3. Busca de video_b.mp4 no n4
.venv\Scripts\python.exe p2p.py config.json animate n1 video_b.mp4 10 flooding busca_video.gif
echo.

echo 4. Random walk
.venv\Scripts\python.exe p2p.py config.json animate n1 archive.zip 4 random_walk random_walk.gif
echo.

echo 5. Busca a partir de n3
.venv\Scripts\python.exe p2p.py config.json animate n3 video_b.mp4 5 flooding busca_n3.gif
echo.

echo Todas as animações foram geradas!
dir *.gif
