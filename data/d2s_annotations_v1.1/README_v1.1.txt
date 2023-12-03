D2S annotations - version 1.1

For the camera ready-version of D2S (ECCV 2018) we changed the following things
compared to d2s_annotations_v1.tar.xz

Class names

suntory_gokuri_limonade --> suntory_gokuri_lemonade
caona_kakaohaltiges_getraenkepulver --> caona_cocoa
cocoba_fruehstueckskakao_mit_honig --> cocoba_cocoa
douwe_egberts_professional_kaffee_gemahlen --> douwe_egberts_professional_ground_coffee
apple_roter_boskoop --> apple_red_boskoop
feldsalat --> corn_salad
rispentomaten --> vine_tomatoes
roma_rispentomaten --> roma_vine_tomatoes
rucola --> rocket

New splits (not mentioned in the paper)

For the validation and test sets we generated new subsets containing specific
difficulties, such as occlusion, clutter or random_background

_clutter --> Images contain a clutter object
_occlusion --> Images contain objects that are touching or occluding each other
_wo_occlusion --> Images where objects are not touching nor occluding each other and don't have a random background
_random_background --> Images with a random_background
_random_background_wo_clutter --> _random_background minus _clutter