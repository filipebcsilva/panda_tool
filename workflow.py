import json
from agno.workflow import Step, Workflow, StepOutput,StepInput
from agents import leitor,saver

def insert_name_image(step_input: StepInput) -> StepOutput:
    """
    Custom function that creates a report using data from multiple previous steps.
    This function has access to ALL previous step outputs and the original workflow message.
    """
    message = step_input.previous_step_content

    name_image = step_input.additional_data.get("image_name")
    
    answers = {
        "answers": {
            name_image: message.answers
        }
    }
    
    answers_str = json.dumps(answers,ensure_ascii=False)
    return StepOutput(
        step_name="insert_name_images", content=answers_str, success=True
    )

leitor_step = Step(
    name = "Leitor",
    agent = leitor
)

saver_step = Step(
    name = "Registrador",
    agent = saver
)

name_image_step = Step(
    name = "Insere nome da imagem",
    executor = insert_name_image
)

vilma_workflow = Workflow(
    name="Pandas pipeline",
    steps=[
        leitor_step,
        name_image_step,
        saver_step
    ],
    debug_mode = True,
)