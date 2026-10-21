-- Staging: the cleaning recipe. Mirrors src/credit_risk/data.py (keep them in sync).
-- RAW stays untouched; this builds a clean copy in CREDIT_RISK.STAGING.

with src as (
    select * from {{ source('raw', 'loans_real') }}
),

cleaned as (
    select
        trim(loanseqnum)                                         as loan_seq_num,

        -- numbers (0 means unknown for credit score and DTI)
        {{ zero_to_null('creditscore') }}                        as credit_score,
        (coalesce(try_to_number(trim(creditscore)), 0) = 0)      as cs_unknown,
        {{ zero_to_null('dti') }}                                as dti,
        (coalesce(try_to_number(trim(dti)), 0) = 0)              as dti_unknown,
        try_to_number(trim(ltv))                                 as ltv,
        try_to_number(trim(origupb))                             as orig_upb,
        try_to_double(trim(originterestrate))                    as orig_interest_rate,
        try_to_number(trim(mip))                                 as mip,
        try_to_number(trim(units))                               as units,
        try_to_number(trim(firstpaymentdate))                    as first_payment_date,

        -- categories ("X" / blank -> UNKNOWN)
        {{ clean_category('firsttimehomebuyer') }}               as first_time_homebuyer,
        {{ clean_category('occupancy') }}                        as occupancy,
        {{ clean_category('channel') }}                          as channel,
        {{ clean_category('ppm') }}                              as ppm,
        {{ clean_category('propertytype') }}                     as property_type,
        {{ clean_category('loanpurpose') }}                      as loan_purpose,
        {{ clean_category('numborrowers') }}                     as num_borrowers,

        -- segments (monitoring / fairness only, never model inputs)
        {{ clean_category('propertystate') }}                    as property_state,
        {{ clean_category('msa') }}                              as msa,
        {{ clean_category('sellername') }}                       as seller_name,

        -- derived
        (try_to_number(trim(ltv)) > 80)                          as ltv_gt80,
        concat(
            left(trim(firstpaymentdate), 4), 'Q',
            ceil(try_to_number(right(trim(firstpaymentdate), 2)) / 3)::int
        )                                                        as vintage_quarter,

        -- outcomes (targets only; never features)
        try_to_number(trim(monthsdelinquent))                    as months_delinquent,
        (try_to_number(trim(monthsdelinquent)) >= 3)::int        as serious_delinquency,
        (try_to_number(trim(monthsdelinquent)) >= 1)::int        as ever_delinquent,

        -- lineage
        _source_file,
        _loaded_at
    from src
)

select *
from cleaned
-- one row per loan
qualify row_number() over (partition by loan_seq_num order by _loaded_at desc) = 1
