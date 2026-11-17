import unittest
from total_shadow import total_quote


class TotalQuoteTest(unittest.TestCase):
    def test_only_same_book_and_fresh_two_sided_total(self):
        quote = dict(captured_at_utc='2026-11-17T13:00:00Z',provider_event_id='e',bookmaker_key='b')
        market = dict(key='totals', last_update='2026-11-17T12:55:00Z',
                      outcomes=[dict(name='Over',point=45),dict(name='Under',point=45)])
        book = dict(key='b',markets=[market])
        events = [dict(id='e',bookmakers=[book])]
        self.assertEqual(total_quote(events,quote)[0],45)
        self.assertIsNone(total_quote(events,dict(quote,bookmaker_key='other')))
        self.assertIsNone(total_quote(events,dict(quote,captured_at_utc='2026-11-17T14:00:00Z')))
        market['outcomes'][0]['point']=46
        self.assertIsNone(total_quote(events,quote))
