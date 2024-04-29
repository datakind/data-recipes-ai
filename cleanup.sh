rm -rf ./ui/recipes_assistant_chat/images/*
rm -rf ./ui/recipes_assistant_chat/logs/*
rm -rf ./ui/recipes_assistant_chat/data-node/*
rm -rf ./ui/recipes_assistant_chat/meili_data_v1.7/*
rm -rf ./ui/recipes_assistant_chat/pgdata2/*
rm -rf ./actions/actions_plugins/recipe-server/db/*

docker compose down -v