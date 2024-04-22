
CREATE SCHEMA data_recipes;
CREATE SCHEMA data;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

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


-- Tabular view onto Langchain vector store collection
CREATE VIEW memory_view as
SELECT 
  le.collection_id,
  le.embedding,
  le.document,
  le.cmetadata,
  le.custom_id,
  le.uuid, 
  le.cmetadata->>'intent' AS intent,
  le.cmetadata->>'response_format' AS response_format,
  le.cmetadata->>'function_response_fields' AS function_response_fields,
  le.cmetadata->>'response_text' AS response_text,
  le.cmetadata->>'response_image' AS response_image,
  le.cmetadata->>'created' AS created,
  le.cmetadata->>'source' AS source, 
  le.cmetadata->>'data_sources' AS data_sources,
  le.cmetadata->>'functions_code' AS functions_code,
  le.cmetadata->>'calling_code' AS calling_code,
  le.cmetadata->>'calling_code_run_status' AS calling_code_run_status,
  le.cmetadata->>'calling_code_run_time' AS calling_code_run_time,
  le.cmetadata->>'mem_type' AS mem_type
FROM 
    langchain_pg_embedding le,
    langchain_pg_collection lc
where 
	le.collection_id = lc.uuid and 
	lc."name" = 'memory_embedding';


CREATE VIEW recipe_view as
SELECT 
  le.collection_id,
  le.embedding,
  le.document,
  le.cmetadata,
  le.custom_id,
  le.uuid, 
  le.cmetadata->>'intent' AS intent,
  le.cmetadata->>'created' AS created,
  le.cmetadata->>'source' AS source,  
  le.cmetadata->>'data_sources' AS data_sources,
  le.cmetadata->>'functions_code' AS functions_code,
  le.cmetadata->>'top_function' AS top_function,
  le.cmetadata->>'response_format' AS response_format,
  le.cmetadata->>'function_response_fields' AS function_response_fields,
  le.cmetadata->>'mem_type' AS mem_type
FROM 
    langchain_pg_embedding le,
    langchain_pg_collection lc
where 
	le.collection_id = lc.uuid and 
	lc."name" = 'recipe_embedding';

CREATE VIEW helper_function_view as
SELECT 
  le.collection_id,
  le.embedding,
  le.document,
  le.cmetadata,
  le.custom_id,
  le.uuid, 
  le.cmetadata->>'intent' AS intent,
  le.cmetadata->>'created' AS created,
  le.cmetadata->>'source' AS source,  
  le.cmetadata->>'data_sources' AS data_sources,
  le.cmetadata->>'functions_code' AS functions_code,
  le.cmetadata->>'top_function' AS top_function,
  le.cmetadata->>'response_format' AS response_format,
  le.cmetadata->>'function_response_fields' AS function_response_fields,
  le.cmetadata->>'mem_type' AS mem_type
FROM 
    langchain_pg_embedding le,
    langchain_pg_collection lc
where 
	le.collection_id = lc.uuid and 
	lc."name" = 'helper_function_embedding';