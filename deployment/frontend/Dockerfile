FROM nginx:1.27-alpine

COPY deployment/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY deployment/frontend/index.html /usr/share/nginx/html/index.html
COPY deployment/frontend/app.js /usr/share/nginx/html/app.js
COPY deployment/frontend/styles.css /usr/share/nginx/html/styles.css

EXPOSE 80
