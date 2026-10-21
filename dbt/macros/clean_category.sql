{# Same rule as data.py: trim; "X" or blank means unknown -> 'UNKNOWN'. #}
{% macro clean_category(col) -%}
    coalesce(nullif(nullif(trim({{ col }}), 'X'), ''), 'UNKNOWN')
{%- endmacro %}

{# Same rule as data.py: 0 means unknown -> NULL. #}
{% macro zero_to_null(col) -%}
    nullif(try_to_number(trim({{ col }})), 0)
{%- endmacro %}
