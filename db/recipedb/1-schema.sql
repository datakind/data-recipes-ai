
-- CREATE SCHEMA data_recipes;
-- CREATE SCHEMA data;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create Langchain tables. Langchain will do this also, but more convenient to have as part of build.

DROP TABLE IF EXISTS public.langchain_pg_collection;
CREATE TABLE public.langchain_pg_collection (
	name varchar NULL,
	cmetadata json NULL,
	uuid uuid NOT NULL,
	CONSTRAINT langchain_pg_collection_pkey PRIMARY KEY (uuid)
);

CREATE TABLE public.langchain_pg_embedding (
	collection_id uuid NULL,
	embedding public.vector NULL,
	"document" varchar NULL,
	cmetadata json NULL,
	custom_id varchar NULL,
	uuid uuid NOT NULL,
	CONSTRAINT langchain_pg_embedding_pkey PRIMARY KEY (uuid)
);

-- public.langchain_pg_embedding foreign keys
ALTER TABLE public.langchain_pg_embedding ADD CONSTRAINT langchain_pg_embedding_collection_id_fkey FOREIGN KEY (collection_id) REFERENCES public.langchain_pg_collection(uuid) ON DELETE CASCADE;

ALTER TABLE langchain_pg_embedding
ADD CONSTRAINT custom_id_unique UNIQUE (custom_id);

CREATE TABLE result_type (
  name varchar NOT NULL PRIMARY KEY,
  description varchar NOT NULL
);

INSERT INTO result_type ("name", "description") VALUES  ('text', 'plain text');
INSERT INTO result_type ("name", "description") VALUES  ('image', 'encoded as text');

CREATE TABLE public.recipe (
  custom_id varchar NOT NULL,
  function_code varchar NOT NULL,
  description varchar NOT NULL,
  openapi_json json NOT NULL,
  datasets varchar NULL,
  python_packages varchar[] NULL,
  used_recipes_list varchar[] NULL,
  sample_call varchar NOT NULL,
  sample_result varchar NOT NULL,
  sample_result_type varchar NOT NULL,
  source varchar NOT NULL,
  created_by varchar NOT NULL,
  updated_by varchar NOT NULL,
  last_updated timestamp NOT NULL,
  approval_status varchar NULL,
  approver varchar NULL,
  approval_latest_update varchar NULL,
  locked_by varchar NULL,
  locked_at varchar NULL,
  CONSTRAINT langchain_pg_embedding_pkey2 PRIMARY KEY (custom_id),
  CONSTRAINT result_type_fkey FOREIGN KEY (sample_result_type) REFERENCES result_type(name),
  CONSTRAINT custom_id_fkey FOREIGN KEY (custom_id) REFERENCES langchain_pg_embedding(custom_id) ON DELETE CASCADE
);

CREATE TABLE public.memory (
  custom_id varchar NOT NULL,
  recipe_custom_id uuid NOT NULL,
  recipe_params json NOT NULL,
  result varchar NOT NULL,
  result_type varchar NOT NULL,
  source varchar NOT NULL,
  created_by varchar NOT NULL,
  updated_by varchar NOT NULL,
  last_updated timestamp NOT NULL,
  attribution varchar NULL,
  CONSTRAINT langchain_pg_embedding_pkey3 PRIMARY KEY (custom_id),
  CONSTRAINT result_type_fkey FOREIGN KEY (result_type) REFERENCES result_type(name),
  CONSTRAINT custom_id_fkey FOREIGN KEY (custom_id) REFERENCES langchain_pg_embedding(custom_id) ON DELETE CASCADE
);


-- Ugrade 
/*
insert into recipe 
select 
  le.uuid, 
  le.cmetadata->>'functions_code' AS functions_code,
  'dummy description, update me',
  '{}'::json,
  ARRAY['HAPI'],
  null,
  null,
  le.cmetadata->>'calling_code',
  case when le.cmetadata->>'response_text' is null then
     le.cmetadata->>'response_image'
  else 
     le.cmetadata->>'response_text'
  end,
  case when le.cmetadata->>'response_text' is null then
     'png'
  else 
     'text'
  end,
  'Recipe Manager',
  'Matt',
  'Matt',
  NOW()
FROM 
    langchain_pg_embedding le,
    langchain_pg_collection lc
where 
	le.collection_id = lc.uuid and
	le.cmetadata->>'calling_code' is not null and
	lc."name" = 'recipe_embedding';

delete from public.langchain_pg_embedding where uuid not in (select uuid from recipe);
*/