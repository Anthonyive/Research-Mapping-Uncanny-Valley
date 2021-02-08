import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
import spacy
from collections import Counter
import numpy as np

# word_embedding_model = models.Transformer('distilbert-base-uncased', max_seq_length=768)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=767, activation_function=nn.Tanh())

# sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

sbert = SentenceTransformer('paraphrase-distilroberta-base-v1')
nlp = spacy.load('en_core_web_lg')


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # heads is how many parts we want to split
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be divisible by heads."

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # QK^T = energy
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, head, head_dim)
        # keys shape: (N, key_len, head, head_dim)
        # enerygy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention(Q,K,V)
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # softmax(...) * V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum: (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_sent_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        pretrained_emb=None
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # here instead of src_vocab_size, I use src_sent_size. I.e., how many sents in a document.
        self.sbert_embedding = nn.Embedding(src_sent_size, embed_size)

        if pretrained_emb.all():
            self.sbert_embedding.weight.requires_grad = False
            self.sbert_embedding.load_state_dict(
                {'weight': torch.from_numpy(pretrained_emb)})
        # also here the max_length changes its meaning from max length of words in a sent to max length of sents in a document.
        self.position_embedding = nn.Embedding(max_length, embed_size)

         

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
         
         
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        out = self.dropout(
            (self.sbert_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_sent_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.sbert_embedding = nn.Embedding(trg_sent_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion,
                          dropout, device) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_sent_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        x = self.dropout((self.sbert_embedding(
            x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_sent_size,
        trg_sent_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=768,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_sent_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_sent_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    text = '''
    Hey there, Reddit… I’m writing this here because I don’t know where else to vent my frustration and let it all out. My name is Charlotte, I’m fifteen years old and I think I am in trouble. I am hiding in the basement while typing this, hoping to escape the fury of the monster upstairs.

I’m really scared and my daddy doesn’t believe me. He never believes me when I tell him that what he did is wrong. He has changed a lot in this past year and he had help with that.

He doesn’t take advice from anyone anymore and that woman enables all his erratic behavior and they drink that thing and start laughing and making stupid noises or animal sounds.

I hate her. I really do.

Since she came into our lives everything changed for the worse. My mommy had a nervous breakdown and she had to go for a while into one of those centers for people who lose control.

It’s not so funny that people call them funny farms. There’s nothing funny about them, you know?

Since they divorced, daddy changed a whole lot. Especially since this woman came into our lives and I am forced to live with her every single day until mommy gets out of the institution.

Daddy doesn’t see her for what she is. A destroyer of homes, a wrecker of families… She is the cause my parents divorced, that much I know.

I may not be the sharpest tool in the shed, but I have a gift. I see people for what they really are. I mean, for what they really are. Sometimes they are just normal people, friendly people. Other times, they are beautiful on the outside but they hide their true ugly self on the inside.

Some of them though are not people. They are monsters hiding in a skin suit. Mimicking a normal person. Taking over their host and manipulating others to do their bidding.

Such is the case with this woman my daddy brought home. She is evil and she scares me a lot. Daddy says that I’m like that because I’m just in that period of my life where I am angry all the time.

He tells me all teens are like that. Angry at their parents, teachers, and society in general. I’m not like that, I swear. I just hate that woman who took my daddy away from me and my mommy.

That vile, twisted and wretched monster. I’ll make her go away tonight. Yes, I will.

Daddy says that he and mommy didn’t get the divorce because of the woman, but because they didn’t get along anymore. The flame was gone, he says.

That’s a lie, I know it for sure. They know each other from work and that’s how it all started. I’m not stupid, I can see and sense things.

I remember that when I first saw her true self, I gasped for air and froze, standing still like a statue. Trembling with fear I saw her disgusting face and it scared me a lot. She gently touched my face with her finger and asked me if everything was alright to which I nodded. She grinned and ruffled my hair like I was some sort of pet.

Sorry for the bad language, but I fucking hate that bitch.

The reason I am hiding here is that I dropped a plate and it smashed into a million bits on the kitchen floor. I was trying to help the monster do the dishes. But at the moment the plate broke, she jumped and she started yelling at me, calling me all kinds of names, bad names that I don’t to repeat here.

She lost control for a moment and that’s when I saw her true face again. She’s just an evil old witch who feeds and prey on weak people like my dad. She pulled a knife and pointed it at me. Then she said that I shouldn’t mention the incident to daddy because it will only get worse and on top of that he won’t believe it, so it was pointless.

You see, I didn’t tell you yet but daddy has some sort of purple cloud hanging above his head whenever they are together. I think he’s been hexed and that’s why he can’t think straight anymore, but I know that if I can make the woman go away or kill her, I’ll have my daddy back with my mommy in no time.

She put a love curse on him and now he can only think about her and not anyone else.

The following part is very hard to write, but I’ll give it a try. After the knife incident, I tried running away but she kept me in place.

Then the kitchen changed to something else. A different world filled with blackness, where the winds were howling in a million different tones and where I heard the cries and screams of people trapped in there.

I started crying and yelling and I called for my mommy, my voice echoing through the darkness. Then I heard her say that no one is going to save me and I saw my mommy in chains at that institution, it was horrible and I was beyond scared out of my mind.

Then I heard beasts howling in the distance and I thought they were coming to get me and I screamed and screamed and I couldn’t get out of that evil place.

The monster witch woman grabbed my arms, bruising them in the process. “This is the place for naughty children like you! If you misbehave again, this is where you’ll end up!”

The beasts were coming ever closer and her grip on my arm was tightening that I thought she would break it. That’s when I screamed again and white light came out of my mouth and eyes, blasting the witch away, hurting her I guess and then I woke up here in the basement.

It’s like someone or something took care of me and instructed me with a plan.

There is a small gas canister here beside because sometimes daddy works here for his car things.

I’m a smoker. I know it’s not healthy but some teens do it right? Like my daddy said, some of us pretend to be misfits and pick up bad habits. I think this might help me in my current situation because I have a lighter.

I will burn the witch while she sleeps, right after daddy goes to work. He starts the night shift in a few hours and I’ll just stay here until I hear him leave. I’m just hoping my phone doesn’t die.

A few hours pass and I look outside the small window from my basement and see it’s already night. I check with my phone to make sure that I’m not being deceived by some external forces and it says 10 PM.

I hear daddy telling the witch goodbye and making kissing sounds. Gross, yuck.

He then leaves for work, the front door of the house closing shut behind him. My heart starts beating like crazy inside my chest and sweat is coming down my temples. I’ve never been more afraid than right now in my entire life.

The bitch witch is sleeping; I can hear her snoring echoing throughout the house.

Good for me.

The gas canister is very small and that makes it easier for me to carry it. I make sure to check the lighter still has gas of its own and it does.

Phew, all good so far.

I slowly open the door to the basement and remove my shoes so I don’t make any unwanted noises. I go upstairs and slowly listen to the witch’s snoring. She seems to talk gibberish in her sleep. Holy hell, I am scared and my hands are sweaty and shaking.

I slowly turn the doorknob, making sure I don’t make a sound. I enter the room and see her. She sleeps in her true form, this evil witch. She’s a monster that needs to be killed. Right here and now.

I douse her in gasoline and she jumps out of bed, screaming and kicking demanding to know what the hell is going on. The bedsheet sticks on her skin, and she can’t seem to free her hands.

“What are you doing, you crazy child?” she screams, her eyes bulging in disbelief.

“I’m just lighting a cigarette. You evil fucking monster,” I tell her, grinning.

I struck the lighter and throw it on the witch. The flames engulf her body and she’s screaming and kicking and I’m scared shitless as she tries to come and catch me but I run outside of the room.

Her dying screams are evaporating in the stillness of the night. I’m crying tears of joy, still not knowing how I managed to pull that off. She’s giving her final breath and I go back to see just a pile of grey and black ash.

I put it in a metal box and bury it in my garden outside. I don’t know how I didn’t burn down the house. The only things that burned were the bedsheet and the witch.

“You’re one of us now. The ones who serve the light. You are so brave and strong,” I hear a voice telling me.

I just shrug it off, saying OK and thanking the voice for their help. The bruise from earlier still hurts, but it will heal soon.

Now I just wait for my daddy to come home and I can’t wait for mommy to get out of the hospital.

I will never forget this experience and I will do anything necessary to protect my family.

    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
    #     device
    # )
    # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # src_pad_idx = 0
    # trg_pad_idx = 0
    # src_vocab_size = 10
    # trg_vocab_size = 10
    # model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
    #     device
    # )
    # out = model(x, trg[:, :-1])
    # print(out.shape)

    doc = nlp(text)
    sents = Counter(doc.sents)  # create a dictionary
    sents = sorted(sents, key=sents.get, reverse=True)
    sents_size = len(sents)
    # x are the indices
    sent2x = {sent: ind for ind, sent in enumerate(doc.sents)}
    sbert_embeddings = sbert.encode([str(sent)
                                     for sent in doc.sents], device=device)
    x = torch.tensor([sent2x[sent] for sent in doc.sents]).to(device)
    x = torch.unsqueeze(x, 1)
     

    src_sent_size, embed_size = sents_size, 768
    # src_sent_size += 10
     

    src_pad_idx = 0
    num_layers = 6
    forward_expansion = 4
    heads = 16
    dropout = 0
    max_length = 100
    src_mask = (x != 0).unsqueeze(1).unsqueeze(2).to(device)
     

    encoder = Encoder(
        src_sent_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        sbert_embeddings
    ).to(device)

     

    out = encoder(x, src_mask)
    print(out, out.shape)
