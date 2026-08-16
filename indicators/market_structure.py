import pandas as pd


class MarketStructureIndicators:


    def __init__(self, dataframe):

        self.df = dataframe.copy()



    def calculate(self):


        # ==========================================
        # SWING DETECTION
        # ==========================================

        window = 5


        self.df["swing_high"] = False

        self.df["swing_low"] = False



        for i in range(
            window,
            len(self.df) - window
        ):


            current_high = self.df["high"].iloc[i]


            current_low = self.df["low"].iloc[i]



            previous_highs = self.df["high"].iloc[
                i-window:i
            ]


            next_highs = self.df["high"].iloc[
                i+1:i+window+1
            ]



            previous_lows = self.df["low"].iloc[
                i-window:i
            ]


            next_lows = self.df["low"].iloc[
                i+1:i+window+1
            ]



            if (

                current_high >

                previous_highs.max()

                and

                current_high >

                next_highs.max()

            ):

                self.df.loc[
                    self.df.index[i],
                    "swing_high"
                ] = True



            if (

                current_low <

                previous_lows.min()

                and

                current_low <

                next_lows.min()

            ):

                self.df.loc[
                    self.df.index[i],
                    "swing_low"
                ] = True



        # ==========================================
        # ÚLTIMOS SWINGS CONFIRMADOS
        # ==========================================


        self.df["last_swing_high"] = None

        self.df["last_swing_low"] = None



        last_high = None

        last_low = None



        for index, row in self.df.iterrows():


            if row["swing_high"]:

                last_high = row["high"]



            if row["swing_low"]:

                last_low = row["low"]



            self.df.loc[
                index,
                "last_swing_high"
            ] = last_high



            self.df.loc[
                index,
                "last_swing_low"
            ] = last_low



        # ==========================================
        # CLASSIFICAÇÃO HH HL LH LL
        # ==========================================


        self.df["higher_high"] = False

        self.df["higher_low"] = False

        self.df["lower_high"] = False

        self.df["lower_low"] = False



        previous_high = None

        previous_low = None



        for index, row in self.df.iterrows():


            if row["swing_high"]:


                if previous_high is not None:


                    if row["high"] > previous_high:

                        self.df.loc[
                            index,
                            "higher_high"
                        ] = True


                    elif row["high"] < previous_high:

                        self.df.loc[
                            index,
                            "lower_high"
                        ] = True



                previous_high = row["high"]




            if row["swing_low"]:


                if previous_low is not None:


                    if row["low"] > previous_low:

                        self.df.loc[
                            index,
                            "higher_low"
                        ] = True


                    elif row["low"] < previous_low:

                        self.df.loc[
                            index,
                            "lower_low"
                        ] = True



                previous_low = row["low"]




        # ==========================================
        # BOS / CHOCH
        # ==========================================


        self.df["bos"] = False

        self.df["choch"] = False



        structure = "NEUTRAL"



        for index, row in self.df.iterrows():


            close = row["close"]



            swing_high = row["last_swing_high"]

            swing_low = row["last_swing_low"]



            if swing_high is not None:


                if close > swing_high:


                    if structure == "BULLISH":


                        self.df.loc[
                            index,
                            "bos"
                        ] = True



                    elif structure == "BEARISH":


                        self.df.loc[
                            index,
                            "choch"
                        ] = True



                    structure = "BULLISH"



            if swing_low is not None:


                if close < swing_low:


                    if structure == "BEARISH":


                        self.df.loc[
                            index,
                            "bos"
                        ] = True



                    elif structure == "BULLISH":


                        self.df.loc[
                            index,
                            "choch"
                        ] = True



                    structure = "BEARISH"



        # ==========================================
        # MARKET STRUCTURE FINAL
        # ==========================================


        result = []

        score = []



        for _, row in self.df.iterrows():


            points = 0



            if row["higher_high"]:

                points += 25


            if row["higher_low"]:

                points += 25


            if row["lower_high"]:

                points -= 25


            if row["lower_low"]:

                points -= 25



            if points > 0:

                result.append(
                    "BULLISH_STRUCTURE"
                )


            elif points < 0:

                result.append(
                    "BEARISH_STRUCTURE"
                )


            else:

                result.append(
                    "NEUTRAL_STRUCTURE"
                )



            score.append(points)



        self.df["market_structure"] = result

        self.df["structure_score"] = score



        return self.df