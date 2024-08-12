document.addEventListener("DOMContentLoaded", function() {
    const clipboardButtons = document.querySelectorAll('.md-clipboard');

    clipboardButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            const codeElement = document.querySelector(button.getAttribute('data-clipboard-target'));
            if (codeElement) {
                let codeToCopy = Array.from(codeElement.childNodes)
                    .filter(node => node.nodeType === Node.TEXT_NODE || (node.nodeType === Node.ELEMENT_NODE && !node.classList.contains('go')))
                    .map(node => node.textContent)
                    .join('')
                    .replace(/^(>>>|\.\.\.)\s/gm, '');

                navigator.clipboard.writeText(codeToCopy).then(() => {
                    console.log("Copied to clipboard successfully!");
                }).catch(err => {
                    console.error("Failed to copy text: ", err);
                });
                event.preventDefault();
            }
        });
    });
});


/* 
* This function was used to remove the python prompt chars i.e. >>> and ... from the 
* copied code. But the above function extends the functionality by also remove the python
* output.
*/
// document.addEventListener("DOMContentLoaded", function() {
//     const clipboardButtons = document.querySelectorAll('.md-clipboard');

//     clipboardButtons.forEach(button => {
//         button.addEventListener('click', function(event) {
//             const codeElement = document.querySelector(button.getAttribute('data-clipboard-target'));
//             if (codeElement) {
//                 let codeToCopy = codeElement.innerText;

//                 // Remove the >>> and ... prompts from the copied text
//                 codeToCopy = codeToCopy.replace(/^>>> /gm, '').replace(/^... /gm, '');

//                 // Copy the cleaned code to clipboard
//                 navigator.clipboard.writeText(codeToCopy).then(() => {
//                     console.log("Copied to clipboard successfully!");
//                 }).catch(err => {
//                     console.error("Failed to copy text: ", err);
//                 });
//                 // Prevent the default copy behavior
//                 event.preventDefault(); 
//             }
//         });
//     });
// });