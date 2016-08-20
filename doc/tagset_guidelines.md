Sanskrit Tagset Guidelines
======

The tagset proposed in this document is a modified (i.e. simplified) variant of the tagset developed by R. Chandrashekar at JNU.
The aim of simplifying the tagset is to reduce the number of possible tags primarily through condensing multiple tags to align more closely with a semantic group. Unsupervised POS-tagging using Baum-Welch tends to maximise tags around semantic groups rather than purely syntactical. Additionally, many of the rarer forms do not appear in any training data, making it impossible for the HMM to learn them.

All credit for developing the tagset goes to R. Chandrashekar. These modified tagset guidelines present his tagset, merely stripped down to a reduced form. The original guidelines can be found at <http://sanskrit.jnu.ac.in/corpora/JNU-Sanskrit-Tagset.htm>.

##Nominals:

###Noun Tags:

Gender sub-tags: `p`,`s`,`n` for puṃliṅga, strīliṅga, napuṃsakaliṅga respectively (masculine, feminine, neuter).

Declensional sub-tags: `vibhakti.vacana` (that is, `case.number`) e.g. 1.1 for prathamā vibhakti, ekavacana, 2.1 dvitīyā vibhakti, ekavacana, etc.

| Tag | Description |
| --- | ----------- |
| **N** | Nāman (Common noun) |
| **NA** | Nāman abhidhāna (Proper noun) |

*Example*: `vegena[N_p_3.1]` (Noun, masculine, 3. vibhakti \[i.e. instrumental case\], singular)

Noun compound tags have been entirely eliminated, and so-called "participles" have been reduced into noun tags since they are declined like any other noun and thus syntactically indistinct.

###Pronoun Tags:

| Tag | Description |
| --- | ----------- |
| **SN** | Sarvanāman (Pronoun) |
| **SNU** | Sarvanāman uttama (First person pronoun) |
| **SNM** | Sarvanāman madhyama (Second person pronoun) |
| **SNN** | Sarvanāman nirdeṣātmaka (Demonstrative pronoun e.g. idam) |
| **SNP** | Sarvanāman prāśnārthika (Interrogative pronoun e.g. kaḥ) |
| **SNS** | Sarvanāman sāmbandhika (Relative pronoun e.g. yaḥ) |
| **SNA** | Sarvanāman ātman (Reflexive pronoun) |

*Example*: `asya[SNN_p_6.1]` (Demonstrative pronoun, masculine, 6. vibhakti \[i.e. genitive\], singular)

###Adjective Tags:

| Tag | Description |
| --- | ----------- |
| **NVI** | Nāma viśeṣaṇa (Adjective) |

*Example*: `viśālāḥ[NVI_p_1.3]` (Adjective, masculine, 1. vibhakti \[nominative\], plural)

###Number Tags:

| Tag | Description |
| --- | ----------- |
| **SAM** | Saṃkhyā (Cardinal number) |

*Example*: `aṣṭau[SAM_p_1.3]` (Cardinal number, masculine, 1. vibhakti, plural)

##Verb Tags:

For simplification (i.e. reduction of total possible tags), verb tagging will not distinguish between parasmai- and ātmanepadas.
Nor is there, for the purposes of this tagging, a distinction between causal, desiderous, nominal and passive verbs.

In addition, the benedictive mood is no longer its own tag due to its rarity (i.e. it is unlikely to appear in any labeled training
data, making it difficult to know where it belongs in unlabeled training data, especially due to its semantic similarity to the optative
or imperative moods). Similarly, the vedic subjunctive and conditional forms are omitted.

Much like for noun tags, following the main verb tag are person and number sub-tags in the form `puruṣa.vacana` (`person.number`).

| Tag | Description |
| --- | ----------- |
| **laTV** | Vartamāna (Present tense) |
| **liTB** | Bhūta (Past perfect tense) |
| **luTAg** | Āgāmī (Periphrastic future) |
| **lRuTAg** | Āgāmī (Future tense) |
| **loTA** | Imperative |
| **la~gB** | Bhūta (Past imperfect tense) |
| **li~gVi** | Vidhyādi (Optative) |
| **lu~gB** | Bhūta (Past aorist tense) |

*Example*: `vahati[laTV_1.1]` (Present tense, first person, singular)

##Avyaya (Particle) tags:

| Tag | Description |
| --- | ----------- |
| **AV**  | Avyaya (particles e.g. atha, iva, saha) |
| **AVN** | Avyaya niṣedhārthaka (e.g. na, mā) |
| **AVC** | Conjunctive avyaya (e.g. ca, tu) |
| **AVD** | Disjunctive avyaya (e.g. vā) |
| **AVP** | Avyaya praśnārthika (interrogatives e.g. api) |
| **AVT** | Avyaya tumunnanta (infinitives e.g. gantum) |
| **AVG** | Avyaya ktvānta/lyabanta (gerunds e.g. gatvā, avalambya) |
| **AVKV** | Avyaya kriyāviśeṣaṇa (adverbs) |
| **AVS** | Avyaya sambodhane (interjection e.g. hā, bho) |

##Punctuation tags:

Many punctuation tags have been condensed into the following two:

| Tag | Description |
| --- | ----------- |
| **PUN\_VV** | Vākya virāma (sentence end, half śloka, question mark, etc.) |
| **PUN\_LV** | Laghu virāma (comma, semicolon) |
| **PUN\_SA** | Ślokānta (śloka end marker) |

##Other tags:

| Tag | Description |
| --- | ----------- |
| **AB** | anyabhāṣā (foreign word) |

