from rasa_sdk import Action
from rasa_sdk.events import SlotSet, EventType
import spacy
import requests


class ActionApplySpacy(Action):
    def __init__(self):
        self.nlp = spacy.load("../spacy/output_model/model-best")
        self.entity_labels = {"style", "objects", "scenes", "colors", "mood", "action"}

    def name(self) -> str:
        return "action_apply_spacy"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")
        doc = self.nlp(user_input)
        extracted_entities = []
        slot_events = []

        for ent in doc.ents:
            label = ent.label_.lower()
            if label in self.entity_labels:
                extracted_entities.append((label, ent.text))
                slot_events.append(SlotSet(label, ent.text))

        if extracted_entities:
            message = "Here’s what I detected from your message:\n"
            for label, text in extracted_entities:
                message += f"- <span style='font-weight: bold'>{label.capitalize()}</span>: {text}\n"
        else:
            message = "I couldn’t detect any relevant entities 😕"

        dispatcher.utter_message(text=f"{message}\n\n")
        return slot_events


class ActionSummarizePrompt(Action):
    def name(self) -> str:
        return "action_summarize_prompt"

    def run(self, dispatcher, tracker, domain) -> list[EventType]:
        # Retrieve slots
        style = tracker.get_slot("style")
        objects = tracker.get_slot("objects")
        scenes = tracker.get_slot("scenes")
        colors = tracker.get_slot("colors")
        mood = tracker.get_slot("mood")
        action = tracker.get_slot("action")

        summary_parts = []

        if style:
            summary_parts.append(
                f"a <span style='font-weight: bold'>{style}<span> style"
            )
        if mood:
            summary_parts.append(
                f"conveying a  <span style='font-weight: bold'>{mood}<span> mood"
            )
        if action:
            summary_parts.append(
                f"with the action  <span style='font-weight: bold'>{action}<span>"
            )
        if scenes:
            summary_parts.append(
                f"in a  <span style='font-weight: bold'>{scenes}<span> setting"
            )
        if colors:
            summary_parts.append(
                f"with  <span style='font-weight: bold'>{colors}<span> hues"
            )
        if objects:
            summary_parts.append(
                f"featuring  <span style='font-weight: bold'>{objects}<span>"
            )

        if summary_parts:
            summary = (
                "✨ <span style='font-weight: bold'>Here’s what I understood from your description:</span>\n\n"
                + "Imagine "
                + ", ".join(summary_parts)
                + ".\n\n"
            )
        else:
            summary = "😕 I wasn't able to extract anything meaningful. Could you try rephrasing?"

        dispatcher.utter_message(text=summary)
        return []


class ActionGenerateImageFromRasa(Action):
    def name(self) -> str:
        return "action_generate_image_from_rasa"

    def run(self, dispatcher, tracker, domain):
        # Récupérer le prompt depuis le slot ou la conversation
        prompt = tracker.get_slot("user_prompt")

        if not prompt:
            dispatcher.utter_message(text="Please provide a prompt for the image.")
            return []

        # URL de la route Flask pour générer l'image
        flask_url = "http://127.0.0.1:5010/auto_submit_prompt"

        try:
            # Envoi de la requête POST avec le prompt vers Flask
            response = requests.post(flask_url, json={"prompt": prompt})

            if response.status_code == 200:
                # Récupérer la réponse JSON
                data = response.json()

                # Récupérer l'image générée depuis la réponse JSON
                img_data = data.get("image")

                if img_data:
                    # Afficher l'image dans la conversation Rasa
                    # Remarquez qu'on ne peut pas envoyer directement un objet base64 avec 'image='
                    # Il faut inclure l'image comme un lien direct ou une URL.
                    dispatcher.utter_message(text="Here is the generated image 👇")
                    dispatcher.utter_message(text="![Generated Image](%s)" % img_data)
                else:
                    dispatcher.utter_message(
                        text="No image generated. Please try a different prompt."
                    )
            else:
                dispatcher.utter_message(
                    text="There was an error generating the image."
                )

        except requests.exceptions.RequestException as e:
            dispatcher.utter_message(text=f"Error contacting Flask server: {str(e)}")

        # Réinitialiser le slot ou continuer avec un autre flux
        return [SlotSet("user_prompt", None)]
