Sanskrit Tagset Guidelines
======

The tagset proposed in this document is a modified (i.e. simplified) variant of the tagset developed by R. Chandrashekar at JNU.
The aim of simplifying the tagset is to reduce the number of possible tags primarily through condensing multiple tags to align more closely with a semantic group. Unsupervised POS-tagging using Baum-Welch tends to maximise tags around semantic groups rather purely syntactical. Additionally, many of the rarer forms do not appear in any training data, making it impossible for the HMM to learn them.

All credit for developing the tagset goes to R. Chandrashekar. These modified tagset guidelines present his tagset, merely stripped down to a reduced form. The original guidelines can be found at <http://sanskrit.jnu.ac.in/corpora/JNU-Sanskrit-Tagset.htm>.

##Noun Tags:

Gender sub-tags: `p`,`s`,`n` (puṃliṅga, strīliṅga, napuṃsakaliṅga)

Declensional sub-tags: `[vibhakti].[vacana]` e.g. 1.1 for prathama vibhakti, ekavacana, 2.1 dvitīya vibhakti, ekavacana, etc.

| Tag | Description |
| --- | ----------- |
| **N** | Nāmapada (Common noun) |
| **NA** | Nāma abhidhāna (Proper noun) |
| **NS** | Nāma sannanta (Desiderative noun) |

*Example*: `vegena[N_p_3.1]` (nāmapada-puṃliṅga-tṛtīyavibhakti-ekavacana)

(Noun compound tags have been entirely eliminated).

###Pronoun Tags:

| Tag | Description |
| --- | ----------- |
| **SN** | Sarvanāman (Pronoun) |
| **SNU** | Sarvanāman uttama (First person pronoun) |
| **SNM** | Sarvanāman madhyama (Second person pronoun) |
| **SNN** | Sarvanāman nirdeṣātmaka (Demonstrative pronoun e.g. idam) |
| **SNP** | Sarvanāman prāśnārthika (Interrogative pronoun e.g. kim) |
| **SNS** | Sarvanāman sāmbandhika (Relative pronoun e.g. yaḥ) |
| **SNA** | Sarvanāman ātman (Reflexive pronoun) |

*Example*: `asya[SNN_p_6.1]` (sarvanāma-puṃliṅga-ṣaṣṭhavibhakti-ekavacana)

###Adjective Tags:

| Tag | Description |
| --- | ----------- |
| **NVI** | Nāma viśeṣaṇa (Adjective) |

*Example*: `viśālāḥ[NVI_p_1.3]` (nāmaviśeṣaṇa-puṃliṅga-prathamavibhakti-bahuvacana)

###Number Tags:

| Tag | Description |
| --- | ----------- |
| **SAM** | Saṃkhyā (Cardinal number) |
| **SAMY** | Saṃkhyeya (Ordinal number) |

*Example*: `aṣṭau[SAMC_p_1.3]` (saṃkhyā-puṃliṅga-prathamavibhakti-bahuvacana)

##Participle Tags:

| Tag | Description |
| --- | ----------- |
| **KV** | Kṛdanta vartamāna 1 (Present active/middle participle) |
| **KB1** | Kṛdanta bhūta 1 (Past passive participle) |
| **KB2** | Kṛdanta bhūta 2 (Past active participle) |
| **KAa** | Kṛdanta āgāmī a (Future active participle) |
| **KAb** | Kṛdanta āgāmī b (Future passive participle) |
| **KVI** | Kṛdanta vidhyārthaka (Gerundive) |

*Example*: `kriyamāṇaḥ[KV_p_1.1]` (vartamāna-puṃliṅga-prathamāvibhakti-ekavacana)

##Verb Tags:

For simplification (i.e. reduction of total possible tags), verb tagging will not distinguish between parasmai- and ātmanepadas.
Nor is there, for the purposes of this tagging, a distinction between causal, desiderous, nominal and passive verbs.

In addition, the benedictive mood is no longer its own tag due to its rarity (i.e. it is unlikely to appear in any labeled training
data, making it difficult to know where it belongs in unlabeled training data, especially due to its semantic similarity to the optative
or imperative moods).

Much like for noun tags, following the main verb tag are person and number sub-tags in the form `[puruṣa].[vacana]`.

| Tag | Description |
| --- | ----------- |
| laTV | Vartamāna (Present tense) |
| liTB | Bhūta (Past perfect tense) |
| luTAg | Āgāmī (Periphrastic future) |
| lRuTAg | Āgāmī (Future tense) |
| loTA | Ājñā (Imperative) |
| la~gB | Bhūta (Past imperfect tense) |
| li~gVi | Vidhi (Optative) |
| lu~gB | Bhūta (Past aorist tense) |

*Example*: `vahati[laTV_1.1]` (laT-lakAra-prathamapuruṣa-ekavacana)

##Avyaya (Particle) tags:

| Tag | Description |
| --- | ----------- |
| AV  | Avyaya (particles e.g. atha, iva, saha) |
| AVN | Avyaya niṣedhārthaka (e.g. na, mā) |
| AVC | Conjunctive avyaya (e.g. ca, tu) |
| AVD | Disjunctive avyaya (e.g. vā) |
| AVP | Avyaya praśnārthika (interrogatives e.g. api) |
| AVT | Avyaya tumunnanta (infinitives e.g. gantum) |
| AVG | Avyaya ktvānta/lyabanta (gerunds e.g. gatvā, avalambya) |
| AVKV | Avyaya kriyāviśeṣaṇa (adverbs) |
| UD | Udgara (interjection e.g. hā, bho) |

##Punctuation tags:

Many punctuation tags have been condensed into the following two:

| Tag | Description |
| --- | ----------- |
| PUN\_VV | Vākya virāma (sentence end, half śloka, comma, question mark, etc.) |
| PUN\_SA | Ślokānta (śloka end marker) |

##Other tags:

| Tag | Description |
| --- | ----------- |
| AB | anyabhāṣā (foreign word) |

