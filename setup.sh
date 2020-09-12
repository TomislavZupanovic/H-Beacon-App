mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"tomislav35_@hotmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
