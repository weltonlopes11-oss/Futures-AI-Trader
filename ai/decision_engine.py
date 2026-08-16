from ai.scoring import ScoreEngine


class DecisionEngine:


    def __init__(self, df):

        self.df = df.copy()

        self.score_engine = ScoreEngine()



    def analyze(self):


        self.df["ai_score"] = (
            self.score_engine
            .calculate(self.df)
        )


        self.df["signal"] = (
            self.generate_signal()
        )


        return self.df



    def generate_signal(self):


        signals = []


        for score in self.df["ai_score"]:


            if score >= 70:

                signals.append(
                    "LONG"
                )


            elif score <= 30:

                signals.append(
                    "SHORT"
                )


            else:

                signals.append(
                    "WAIT"
                )


        return signals