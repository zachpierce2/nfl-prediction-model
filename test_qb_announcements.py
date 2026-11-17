import json
import unittest
from datetime import datetime, timezone
from qb_announcements import article, interpret, upcoming, reconcile, qb_players, combine_confirmations

NOW=datetime(2026,11,17,18,tzinfo=timezone.utc)
URL='https://www.clevelandbrowns.com/news/decision'
GAME=dict(game_id='2026_11_HOU_CLE',season='2026',week='11',game_type='REG',home_team='CLE',away_team='HOU',gameday='2026-11-22',gametime='13:00',kickoff_utc='2026-11-22T18:00:00Z')
PLAYER=dict(gsis_id='00-0033537',display_name='Deshaun Watson',position='QB',latest_team='CLE',last_season='2026')
TITLE='Deshaun Watson set to start Week 11 against the Texans'


def sample(title=TITLE,body=None):
    return dict(title=title,body=body or title,team='CLE',source_url=URL,
        published_at_utc='2026-11-17T17:00:00Z',observed_at_utc=NOW.isoformat())


class AnnouncementTests(unittest.TestCase):
    def decisions(self,a):
        return interpret(a,[GAME],[PLAYER])

    def test_explicit_game_maps_to_gsis(self):
        r=self.decisions(sample())
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]['confirmation']['qb_id'],PLAYER['gsis_id'])
        self.assertEqual(r[0]['confirmation']['game_id'],GAME['game_id'])
        self.assertEqual(r[0]['confirmation']['status'],'confirmed')

    def test_no_week_opponent_or_wrong_year_abstains(self):
        for title in [TITLE.replace('Week 11','Week 10'),TITLE.replace('Texans','Bengals'),
            'Deshaun Watson named starting quarterback',TITLE+' in 2027',TITLE.replace('Week 11','')]:
            with self.subTest(title=title):self.assertEqual(self.decisions(sample(title)),[])

    def test_speculation_question_condition_and_history_abstain(self):
        for title in [TITLE+'?',TITLE+' if healthy',TITLE.replace('set to start','expected to start'),
            TITLE+' barring a setback',TITLE+' last year','Coach: "'+TITLE+'"']:
            with self.subTest(title=title):self.assertEqual(self.decisions(sample(title)),[])

    def test_identity_team_season_or_ambiguity(self):
        for p in [dict(PLAYER,latest_team='BUF'),dict(PLAYER,last_season='2025')]:
            self.assertEqual(interpret(sample(),[GAME],[p]),[])
        # Ambiguous IDs yield unknown when reconciled, never an arbitrary identity.
        rows=interpret(sample(),[GAME],[PLAYER,dict(PLAYER,gsis_id='00-0000001')])
        self.assertEqual(reconcile([],rows,{URL},NOW)[0]['status'],'unknown')

    def test_late_observation_and_future_publication(self):
        a=sample();a['observed_at_utc']='2026-11-22T18:00:00Z'
        self.assertEqual(self.decisions(a),[])
        a=sample();a['published_at_utc']='2026-11-18T00:00:00Z'
        self.assertEqual(self.decisions(a),[])

    def test_explicit_retraction(self):
        a=sample(TITLE.replace('set to start','will not start'))
        self.assertEqual(self.decisions(a)[0]['confirmation']['status'],'unknown')

    def test_other_sentence_does_not_provide_game(self):
        a=sample('Deshaun Watson will start. The Browns play Week 11 against the Texans.')
        self.assertEqual(self.decisions(a),[])

    def test_old_reobserved_report_cannot_override_newer(self):
        d=self.decisions(sample())[0]
        old=dict(d['confirmation'],qb_id='00-0000002',published_at_utc='2026-11-17T17:30:00Z',source_url=URL+'-newer')
        self.assertEqual(reconcile([old],[d],{URL},NOW),[])

    def test_repeated_source_does_not_change_first_observation(self):
        d=self.decisions(sample())[0]
        self.assertEqual(reconcile([d['confirmation']],[d],{URL},NOW),[])

    def test_missing_or_removed_source_revokes(self):
        d=self.decisions(sample())[0]['confirmation']
        r=reconcile([d],[],{URL},NOW)
        self.assertEqual(r[0]['status'],'unknown')
        self.assertEqual(r[0]['observed_at_utc'],NOW.isoformat())
        # A previously revoked source can restore only on a new successful check.
        self.assertEqual(reconcile([d,r[0]],self.decisions(sample()),{URL},NOW)[0]['status'],'confirmed')

    def test_article_schema_origin_and_dates(self):
        obj=dict({'@type':'NewsArticle'},url=URL,headline=TITLE,articleBody=TITLE,
            datePublished='2026-11-17T16:00:00Z',dateModified='2026-11-17T17:00:00Z')
        def raw(o):return ('<script type="application/ld+json">'+json.dumps(o)+'</script>').encode()
        a=article(raw(obj),URL,'CLE',NOW)
        self.assertTrue(a['published_at_utc'].startswith('2026-11-17T17:00'))
        for o in [dict(obj,url='https://evil.example/news'),dict(obj,dateModified='2026-11-18T00:00:00Z'),dict(obj,articleBody='')]:
            with self.assertRaises(ValueError):article(raw(o),URL,'CLE',NOW)
        with self.assertRaises(ValueError):article(raw(obj)*2,URL,'CLE',NOW)

    def test_schedule_all_games_and_timezone(self):
        r=upcoming([GAME,dict(GAME,game_id='other',game_type='POST')],NOW)
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]['kickoff_utc'],'2026-11-22T18:00:00+00:00')
        self.assertEqual(upcoming([GAME],datetime(2026,11,22,18,tzinfo=timezone.utc)),[])

    def test_current_qb_gsis_identity_filter(self):
        self.assertEqual(qb_players([PLAYER,dict(PLAYER,position='WR'),dict(PLAYER,gsis_id='bad')],2026),[PLAYER])

    def test_manual_revocation_cannot_be_overridden_by_automatic_observation(self):
        automatic=self.decisions(sample())[0]['confirmation']
        manual=dict(automatic,status='unknown',observed_at_utc='2026-11-17T16:00:00Z')
        self.assertEqual(combine_confirmations([manual],[automatic]),[manual])
        self.assertEqual(combine_confirmations([],[automatic]),[automatic])

    def test_verified_team_week_season_designation(self):
        text="Deshaun Watson will be the Browns' starting quarterback heading into Week 11 of the 2026 season."
        self.assertEqual(self.decisions(sample(text))[0]['confirmation']['qb_id'],PLAYER['gsis_id'])
        for wrong in [text.replace('Week 11','Week 10'),text.replace('2026','2025'),
            text.replace('Browns','Raiders'),text+' if healthy',text.replace(' of the 2026 season','')]:
            self.assertEqual(self.decisions(sample(wrong)),[])

    def test_team_week_designation_requires_unique_game(self):
        text="Deshaun Watson will be the Browns' starting quarterback heading into Week 11 of the 2026 season."
        self.assertEqual(interpret(sample(text),[GAME,dict(GAME,game_id='duplicate')],[PLAYER]),[])


if __name__=='__main__':unittest.main()
