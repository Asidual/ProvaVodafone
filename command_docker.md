Per attivare: docker compose up -d
Per togliere: docker compose down

BE http://localhost:8000
FE http://localhost:8501

docker compose down -v --remove-orphans (elimina i vecchi)

docker compose build --no-cache backend
docker compose build --no-cache frontend
docker compose build --no-cache 

Quale stai lanciando: docker compose ps
