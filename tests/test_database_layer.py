from data.database import Database



db = Database()


db.create_tables()



analysis = {


    "market_state":"RANGE",

    "bias":"WAIT",

    "confidence":60,

    "reasons":[

        "Mercado lateral sem confirmação estrutural"

    ]

}



db.save_market_analysis(

    analysis

)



signal = {


    "signal":"WAIT",

    "long_score":42,

    "short_score":38,

    "confidence":42

}



db.save_signal(

    signal

)



print(

    "\nBanco Analytics Layer funcionando!"

)



db.close()