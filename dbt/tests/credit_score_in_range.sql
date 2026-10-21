-- Fails (returns rows) if a known credit score is outside 300-850.
select loan_seq_num, credit_score
from {{ ref('stg_loans') }}
where credit_score is not null and (credit_score < 300 or credit_score > 850)
