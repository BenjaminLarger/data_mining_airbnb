SHELL	= /bin/sh

NAME	= data_mining

all:
	cd srcs && docker compose up --build

# Remove zone identifier files
delete_zone_identifier:
	find . -name "*Zone.Identifier" -type f -delete

down:
	cd srcs && docker compose down -v
stop:
	cd srcs && docker compose stop
logs:
	cd srcs && docker-compose logs -f

nginx:
	docker exec -it nginx /bin/sh

nginx_restart:
	docker restart nginx

backend:
	docker exec -it backend /bin/sh

restart_backend:
	docker restart backend

.phony: all down stop logs nginx nginx_restart backend restart_backend
