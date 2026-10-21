-- Fails if LTV or DTI is impossible.
select loan_seq_num, ltv, dti
from {{ ref('stg_loans') }}
where (ltv is not null and (ltv < 0 or ltv > 200))
   or (dti is not null and (dti < 0 or dti > 100))
