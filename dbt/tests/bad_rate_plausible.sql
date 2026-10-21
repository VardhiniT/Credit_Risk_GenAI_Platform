-- Fails if the serious-delinquency rate is wildly off (data went wrong). Expected ~9%.
select avg(serious_delinquency) as bad_rate
from {{ ref('stg_loans') }}
having avg(serious_delinquency) not between 0.03 and 0.20
